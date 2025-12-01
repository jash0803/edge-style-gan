import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple
import copy
import sys
import os

# Import quantization patch for Python 3.14 compatibility
try:
    from core.quantization_patch import apply_python314_quantization_patch, patch_quantizer_module
except ImportError:
    # If patch module doesn't exist, create no-op functions
    def apply_python314_quantization_patch():
        pass
    def patch_quantizer_module():
        pass

# Lazy import for quantization to avoid Python 3.14 compatibility issues
_QUANTIZATION_AVAILABLE = None
_quantization_module = None
_prepare_fx = None
_convert_fx = None
_get_default_qconfig_mapping = None

def _patch_pytorch_quantization():
    """Patch PyTorch quantization to work with Python 3.14"""
    # Only patch on Python 3.14+
    if sys.version_info >= (3, 14):
        try:
            # The issue: torch.ao.quantization.quantizer.quantizer line 85 tries to set
            # __module__ on typing.Union, which fails in Python 3.14
            # Solution: Patch typing.Union to allow __module__ assignment
            
            import types
            import typing
            from typing import get_origin
            
            # Check if we're dealing with a Union type that can't have __module__ set
            if hasattr(typing, 'Union'):
                # Create a descriptor that allows __module__ to be "set" (but doesn't actually store it)
                class ModuleDescriptor:
                    """Descriptor that allows __module__ to be set on Union types"""
                    def __get__(self, obj, objtype=None):
                        # Return a fake module name if accessed
                        return "torch.ao.quantization.quantizer.quantizer"
                    
                    def __set__(self, obj, value):
                        # Silently accept the assignment but don't store it
                        # This prevents the AttributeError
                        pass
                
                # Try to set the descriptor on Union
                try:
                    # Use __dict__ if available, otherwise try direct assignment
                    if hasattr(typing.Union, '__dict__'):
                        typing.Union.__dict__['__module__'] = ModuleDescriptor()
                    else:
                        # Try to use setattr with a custom handler
                        # Create a wrapper that intercepts __module__ assignment
                        original_union = typing.Union
                        
                        # Create a new class that behaves like Union but allows __module__
                        class UnionWrapper:
                            """Wrapper for Union that allows __module__ assignment"""
                            def __init__(self):
                                self._origin = original_union
                            
                            def __instancecheck__(self, instance):
                                return isinstance(instance, self._origin)
                            
                            def __subclasscheck__(self, subclass):
                                return issubclass(subclass, self._origin)
                            
                            def __getattr__(self, name):
                                if name == '__module__':
                                    return "torch.ao.quantization.quantizer.quantizer"
                                return getattr(self._origin, name)
                            
                            def __setattr__(self, name, value):
                                if name in ['_origin', '__module__']:
                                    # Allow setting these
                                    object.__setattr__(self, name, value)
                                else:
                                    # For other attributes, try to set on the original
                                    try:
                                        setattr(self._origin, name, value)
                                    except (TypeError, AttributeError):
                                        object.__setattr__(self, name, value)
                        
                        # This approach won't work because we can't replace typing.Union
                        # Instead, we'll patch the module after it's imported
                        pass
                except (TypeError, AttributeError):
                    pass
        except Exception:
            # If patching fails, we'll catch the error during import and use dynamic quantization
            pass


def _try_import_quantization():
    """Try to import quantization modules, handle compatibility issues"""
    global _QUANTIZATION_AVAILABLE, _quantization_module, _prepare_fx, _convert_fx, _get_default_qconfig_mapping
    
    if _QUANTIZATION_AVAILABLE is not None:
        return _QUANTIZATION_AVAILABLE
    
    # Apply Python 3.14 patch BEFORE importing quantization modules
    if sys.version_info >= (3, 14):
        apply_python314_quantization_patch()
    
    # Try to patch PyTorch quantization for Python 3.14 compatibility
    _patch_pytorch_quantization()
    
    try:
        import torch.quantization as quant
        _quantization_module = quant
        _QUANTIZATION_AVAILABLE = True
        
        # Try to import FX-based quantization (may fail on Python 3.14)
        try:
            # Apply patch again right before importing FX modules
            if sys.version_info >= (3, 14):
                patch_quantizer_module()
            _patch_pytorch_quantization()
            
            from torch.ao.quantization import get_default_qconfig_mapping as get_qconfig
            from torch.ao.quantization.quantize_fx import prepare_fx as prep_fx, convert_fx as conv_fx
            
            _prepare_fx = prep_fx
            _convert_fx = conv_fx
            _get_default_qconfig_mapping = get_qconfig
        except AttributeError as e:
            # This is likely the Python 3.14 __module__ error
            if "'typing.Union' object has no attribute '__module__'" in str(e) or "__module__" in str(e):
                # Try to patch the module and retry
                if sys.version_info >= (3, 14):
                    patch_quantizer_module()
                    # Try importing again
                    try:
                        from torch.ao.quantization import get_default_qconfig_mapping as get_qconfig
                        from torch.ao.quantization.quantize_fx import prepare_fx as prep_fx, convert_fx as conv_fx
                        
                        _prepare_fx = prep_fx
                        _convert_fx = conv_fx
                        _get_default_qconfig_mapping = get_qconfig
                    except Exception:
                        # Still failed, FX quantization not available
                        _prepare_fx = None
                        _convert_fx = None
                        _get_default_qconfig_mapping = None
                else:
                    raise
            else:
                # Other error, FX quantization not available
                _prepare_fx = None
                _convert_fx = None
                _get_default_qconfig_mapping = None
        except (ImportError, TypeError) as e:
            # FX quantization not available, but dynamic quantization might work
            _prepare_fx = None
            _convert_fx = None
            _get_default_qconfig_mapping = None
        
        return True
    except (ImportError, AttributeError, TypeError) as e:
        _QUANTIZATION_AVAILABLE = False
        return False


class ModelCompressor:
    """Handles quantization and pruning of MobileStyleGAN models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = None
        self.pruning_masks = {}
        self.sparse_format = None
    
    def save_original_state(self):
        """Save original model state before compression"""
        self.original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def restore_original_state(self):
        """Restore original model state"""
        if self.original_state:
            self.model.load_state_dict(self.original_state)
    
    # ========== QUANTIZATION METHODS ==========
    
    def quantize_static(self, example_inputs: tuple, backend='fbgemm'):
        """
        Static quantization (Post-Training Quantization)
        Best for: Already trained models, no retraining needed
        
        Args:
            example_inputs: Example inputs for calibration
            backend: 'fbgemm' (CPU) or 'qnnpack' (mobile)
        
        Returns:
            Quantized model
        """
        if not _try_import_quantization():
            raise RuntimeError(
                "Quantization not available. This may be due to:\n"
                "1. PyTorch version incompatibility (especially with Python 3.14)\n"
                "2. Missing quantization dependencies\n"
                "Try using dynamic quantization or pruning instead."
            )
        
        # Check available engines and auto-select if needed
        try:
            available_engines = torch.backends.quantized.supported_engines
            if backend not in available_engines:
                # Auto-select available backend
                if 'qnnpack' in available_engines:
                    print(f"Warning: {backend} not available. Using qnnpack instead.")
                    backend = 'qnnpack'
                elif 'fbgemm' in available_engines:
                    print(f"Warning: {backend} not available. Using fbgemm instead.")
                    backend = 'fbgemm'
                else:
                    raise RuntimeError(
                        f"No quantization engines available. Supported: {available_engines}\n"
                        "Try using dynamic quantization or pruning instead."
                    )
        except Exception as e:
            print(f"Warning: Could not check available engines: {e}")
        
        # Initialize quantization backend engine
        try:
            # Set the default quantization engine
            torch.backends.quantized.engine = backend
            print(f"Using quantization engine: {backend}")
        except Exception as e:
            print(f"Warning: Could not set quantization engine: {e}")
            # Try to continue anyway - some backends might work without explicit setting
        
        # Check if FX quantization is available
        if _prepare_fx is None or _convert_fx is None or _get_default_qconfig_mapping is None:
            print("FX-based static quantization not available. Falling back to dynamic quantization...")
            return self.quantize_dynamic()
        
        try:
            # Set quantization config
            qconfig_mapping = _get_default_qconfig_mapping(backend)
            
            # Prepare model for quantization
            prepared_model = _prepare_fx(
                self.model,
                qconfig_mapping,
                example_inputs
            )
            
            # Calibrate with example inputs (can use validation set)
            prepared_model.eval()
            with torch.no_grad():
                _ = prepared_model(*example_inputs)
            
            # Convert to quantized model
            quantized_model = _convert_fx(prepared_model)
            
            return quantized_model
        except RuntimeError as e:
            error_str = str(e)
            if "NoQEngine" in error_str or "engine" in error_str.lower() or "linear_prepack" in error_str.lower():
                # Quantization engine not available - try dynamic quantization
                print(f"Static quantization engine error: {e}")
                print("Falling back to dynamic quantization (weights only)...")
                return self.quantize_dynamic()
            elif "not supported" in error_str.lower() or "not available" in error_str.lower():
                # Backend not supported - try the other one or dynamic
                print(f"Backend {backend} not supported: {e}")
                # Try qnnpack if fbgemm failed, or vice versa
                if backend == 'fbgemm':
                    print("Trying qnnpack backend instead...")
                    try:
                        torch.backends.quantized.engine = 'qnnpack'
                        return self.quantize_static(example_inputs, backend='qnnpack')
                    except Exception:
                        print("qnnpack also failed, using dynamic quantization...")
                        return self.quantize_dynamic()
                else:
                    print("Falling back to dynamic quantization...")
                    return self.quantize_dynamic()
            else:
                raise
        except (AttributeError, TypeError) as e:
            # Python 3.14 compatibility issue - fall back to dynamic quantization
            print(f"Static quantization failed due to compatibility issue: {e}")
            print("Falling back to dynamic quantization...")
            return self.quantize_dynamic()
    
    def quantize_dynamic(self, dtype=torch.qint8):
        """
        Dynamic quantization (weights quantized, activations in float)
        Best for: Models with many Linear layers
        
        Args:
            dtype: Quantization dtype (torch.qint8 or torch.float16)
        
        Returns:
            Quantized model
        """
        if not _try_import_quantization():
            raise RuntimeError(
                "Quantization not available. This may be due to:\n"
                "1. PyTorch version incompatibility (especially with Python 3.14)\n"
                "2. Missing quantization dependencies\n"
                "Try using pruning instead."
            )
        
        # Quantize only linear layers (mapping network and modulation layers)
        quantized_model = _quantization_module.quantize_dynamic(
            self.model,
            {nn.Linear},  # Only quantize Linear layers
            dtype=dtype
        )
        return quantized_model
    
    def quantize_qat(self, example_inputs: tuple, backend='fbgemm'):
        """
        Quantization-Aware Training setup
        Best for: Best accuracy, requires retraining
        
        Args:
            example_inputs: Example inputs for preparation
            backend: Quantization backend
        
        Returns:
            Prepared model for QAT training
        """
        if not _try_import_quantization():
            raise RuntimeError(
                "Quantization not available. This may be due to:\n"
                "1. PyTorch version incompatibility (especially with Python 3.14)\n"
                "2. Missing quantization dependencies\n"
                "Try using pruning instead."
            )
        
        qconfig_mapping = _get_default_qconfig_mapping(backend)
        
        # Prepare model for QAT
        prepared_model = _prepare_fx(
            self.model,
            qconfig_mapping,
            example_inputs
        )
        
        return prepared_model
    
    # ========== PRUNING METHODS ==========
    
    def _get_parameters_to_prune(self):
        """Get all parameters that can be pruned (weights only)"""
        parameters_to_prune = []
        seen_modules = set()
        
        # Find all modules with weight parameters
        for name, module in self.model.named_modules():
            # Skip if already processed
            if id(module) in seen_modules:
                continue
            
            # Skip certain module types that shouldn't be pruned
            skip_types = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, 
                         nn.Embedding, nn.Parameter, type(None))
            if isinstance(module, skip_types):
                continue
            
            # Standard PyTorch layers
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                    parameters_to_prune.append((module, 'weight'))
                    seen_modules.add(id(module))
            
            # Custom modules - check for weight parameters directly
            # This catches ModulatedConv2d, StyledConv2d, etc.
            elif hasattr(module, 'weight'):
                weight = getattr(module, 'weight', None)
                if isinstance(weight, nn.Parameter):
                    # Only prune if it's a multi-dimensional weight (not bias-like)
                    # Skip 1D parameters (likely bias) and scalars
                    if weight.dim() >= 2:
                        # Make sure it's not a bias parameter
                        if not hasattr(module, 'bias') or weight is not getattr(module, 'bias', None):
                            parameters_to_prune.append((module, 'weight'))
                            seen_modules.add(id(module))
        
        return parameters_to_prune
    
    def prune_unstructured(self, amount: float = 0.2, method='magnitude'):
        """
        Unstructured pruning - removes individual weights
        
        Args:
            amount: Fraction of weights to prune (0.0 to 1.0)
            method: 'magnitude' or 'random'
        
        Returns:
            Pruned model
        """
        parameters_to_prune = self._get_parameters_to_prune()
        
        if len(parameters_to_prune) == 0:
            print("Warning: No parameters found to prune")
            return self.model
        
        # Apply pruning
        if method == 'magnitude':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        elif method == 'random':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount,
            )
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Make pruning permanent
        for module, name in parameters_to_prune:
            try:
                prune.remove(module, name)
            except Exception as e:
                # Some modules might not support remove, skip them
                print(f"Warning: Could not remove pruning from {module}: {e}")
        
        return self.model
    
    def prune_structured(self, amount: float = 0.2, dim: int = 0):
        """
        Structured pruning - zeros entire channels/filters
        
        ⚠️  IMPORTANT LIMITATION:
        PyTorch's structured pruning zeros channels but does NOT physically remove them.
        This means the model size on disk stays the same, but inference can be faster
        (zero channels can be skipped during computation).
        
        To actually reduce file size, use:
        - Unstructured pruning + sparse format (--sparse-format coo)
        - Or manually reconstruct the model with fewer channels
        
        Args:
            amount: Fraction of channels to prune
            dim: Dimension to prune (0 for output channels, 1 for input)
        
        Returns:
            Pruned model (channels zeroed, but not removed)
        """
        print("⚠️  WARNING: PyTorch structured pruning zeros channels but doesn't remove them.")
        print("   Model file size will NOT be reduced.")
        print("   For size reduction, use: unstructured pruning + --sparse-format")
        
        pruned_count = 0
        skipped_count = 0
        
        for name, module in self.model.named_modules():
            # Standard Conv2d layers
            if isinstance(module, nn.Conv2d):
                try:
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=amount,
                        n=2,  # L2 norm
                        dim=dim
                    )
                    # Make permanent (zeros channels but keeps them)
                    prune.remove(module, 'weight')
                    pruned_count += 1
                except Exception as e:
                    print(f"Warning: Could not prune {name}: {e}")
                    skipped_count += 1
            
            # Custom modules with Conv2d-like weights (ModulatedConv2d, etc.)
            elif hasattr(module, 'weight') and isinstance(getattr(module, 'weight', None), nn.Parameter):
                weight = getattr(module, 'weight')
                # Only prune if it's a 4D tensor (Conv2d-like)
                if weight.dim() == 4:
                    try:
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=amount,
                            n=2,  # L2 norm
                            dim=dim
                        )
                        prune.remove(module, 'weight')
                        pruned_count += 1
                    except Exception as e:
                        skipped_count += 1
        
        print(f"Structured pruning: {pruned_count} modules pruned, {skipped_count} skipped")
        print("Note: Channels are zeroed but not removed. Model size unchanged.")
        return self.model
    
    def prune_iterative(self, 
                       amounts: List[float] = [0.1, 0.2, 0.3, 0.4],
                       method='magnitude',
                       retrain_fn=None):
        """
        Iterative pruning - gradually increase sparsity
        
        Args:
            amounts: List of pruning amounts to apply iteratively
            method: Pruning method
            retrain_fn: Function to retrain model between pruning steps
        
        Returns:
            Pruned model
        """
        for i, amount in enumerate(amounts):
            print(f"Iterative pruning step {i+1}/{len(amounts)}: {amount*100:.1f}%")
            
            if i == 0:
                # First step: prune from original
                self.prune_unstructured(amount=amount, method=method)
            else:
                # Subsequent steps: prune from already pruned model
                current_sparsity = sum(amounts[:i])
                new_sparsity = sum(amounts[:i+1])
                # Prune additional amount
                additional = (new_sparsity - current_sparsity) / (1 - current_sparsity)
                self.prune_unstructured(amount=additional, method=method)
            
            # Retrain if function provided
            if retrain_fn:
                print("Retraining after pruning...")
                retrain_fn(self.model)
        
        return self.model
    
    def get_model_size(self, model=None):
        """Calculate model size in MB"""
        if model is None:
            model = self.model
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb
    
    def get_sparsity(self):
        """Calculate current model sparsity"""
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.nelement()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity
    
    def get_num_parameters(self):
        """Get total number of parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
    
    def convert_to_sparse(self, format='coo'):
        """
        Convert pruned model to sparse format to reduce file size
        Only works for unstructured pruning (structured pruning already reduces size)
        
        Note: Sparse tensors can't be used directly in forward pass for all operations.
        This is mainly for storage efficiency. For inference, you'd need to convert back.
        
        Args:
            format: Sparse format ('coo', 'csr', 'csc')
                   Note: PyTorch sparse formats have limitations
        
        Returns:
            Model (sparse conversion is stored for checkpoint saving)
        """
        # Store sparse format info for later checkpoint saving
        # We don't actually convert the model parameters to sparse here
        # because sparse tensors can't be nn.Parameters and may break forward pass
        self.sparse_format = format
        print(f"Sparse format set to: {format}")
        print("Note: Sparse conversion will be applied during checkpoint saving")
        return self.model
    
    def save_sparse_checkpoint(self, filepath, format='coo'):
        """
        Save model in sparse format to reduce file size
        This creates a custom checkpoint format that stores sparse tensors efficiently
        
        Args:
            filepath: Path to save the checkpoint
            format: Sparse format to use ('coo', 'csr', 'csc')
        
        Returns:
            Tuple of (sparse_size_mb, dense_size_mb)
        """
        sparse_state_dict = {}
        metadata = {
            'sparse_format': True,
            'format': format,
            'original_params': {},
            'sparse_params': {}
        }
        
        total_dense_size = 0
        total_sparse_size = 0
        
        for name, param in self.model.named_parameters():
            if isinstance(param, torch.Tensor) and param.dim() >= 2:
                # Check sparsity
                sparsity = (param == 0).sum().item() / param.numel()
                total_dense_size += param.numel() * param.element_size()
                
                if sparsity > 0.3:  # Convert if >30% sparse
                    try:
                        # Convert to sparse format
                        if format == 'coo':
                            sparse_param = param.detach().to_sparse_coo()
                        elif format == 'csr' and param.dim() == 2:
                            sparse_param = param.detach().to_sparse_csr()
                        elif format == 'csc' and param.dim() == 2:
                            sparse_param = param.detach().to_sparse_csc()
                        else:
                            # Fallback to COO
                            sparse_param = param.detach().to_sparse_coo()
                        
                        # Calculate sparse size (indices + values)
                        if sparse_param.is_sparse:
                            indices_size = sparse_param.indices().numel() * sparse_param.indices().element_size()
                            values_size = sparse_param.values().numel() * sparse_param.values().element_size()
                            total_sparse_size += indices_size + values_size
                            
                            sparse_state_dict[name] = sparse_param
                            metadata['sparse_params'][name] = {
                                'format': format,
                                'shape': list(param.shape),
                                'sparsity': sparsity,
                                'nnz': sparse_param._nnz() if hasattr(sparse_param, '_nnz') else None
                            }
                        else:
                            # Conversion failed, save as dense
                            sparse_state_dict[name] = param
                            metadata['original_params'][name] = {'format': 'dense', 'shape': list(param.shape)}
                    except Exception as e:
                        # Can't convert, save as dense
                        sparse_state_dict[name] = param
                        metadata['original_params'][name] = {'format': 'dense', 'shape': list(param.shape)}
                else:
                    # Not sparse enough, save as dense
                    sparse_state_dict[name] = param
                    metadata['original_params'][name] = {'format': 'dense', 'shape': list(param.shape)}
            else:
                # Save non-tensor parameters or 1D tensors as-is
                sparse_state_dict[name] = param
        
        # Save checkpoint
        checkpoint = {
            'state_dict': sparse_state_dict,
            'metadata': metadata
        }
        
        torch.save(checkpoint, filepath)
        
        # Calculate actual file size
        sparse_file_size = os.path.getsize(filepath) / (1024 ** 2)
        dense_size_mb = total_dense_size / (1024 ** 2)
        
        return sparse_file_size, dense_size_mb
    
    @staticmethod
    def load_sparse_checkpoint(filepath, map_location='cpu'):
        """
        Load sparse checkpoint and convert back to dense format
        
        This function loads a sparse checkpoint (saved with save_sparse_checkpoint)
        and converts all sparse tensors back to dense format so they can be used
        for model loading and inference.
        
        Args:
            filepath: Path to sparse checkpoint file
            map_location: Device to load checkpoint on ('cpu', 'cuda', etc.)
        
        Returns:
            Dictionary with:
            - 'state_dict': Dense state_dict ready for model.load_state_dict()
            - 'metadata': Original checkpoint metadata
            - All other keys from the original checkpoint
        """
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        
        # Check if this is actually a sparse checkpoint
        is_sparse = isinstance(checkpoint, dict) and checkpoint.get('metadata', {}).get('sparse_format', False)
        
        if not is_sparse:
            # Not a sparse checkpoint, return as-is
            return checkpoint
        
        print("📦 Loading sparse checkpoint and converting to dense format...")
        
        # Get the sparse state dict
        sparse_state_dict = checkpoint.get('state_dict', {})
        metadata = checkpoint.get('metadata', {})
        sparse_params_info = metadata.get('sparse_params', {})
        
        # Convert sparse tensors to dense
        dense_state_dict = {}
        converted_count = 0
        total_params = 0
        
        for name, param in sparse_state_dict.items():
            if isinstance(param, torch.Tensor) and param.is_sparse:
                # Convert sparse tensor back to dense
                try:
                    dense_param = param.to_dense()
                    dense_state_dict[name] = dense_param
                    converted_count += 1
                    
                    # Get info about the conversion
                    if name in sparse_params_info:
                        info = sparse_params_info[name]
                        shape = info.get('shape', list(dense_param.shape))
                        sparsity = info.get('sparsity', 0.0)
                        nnz = info.get('nnz', None)
                        total_params += dense_param.numel()
                        
                        if converted_count <= 3:  # Print first 3 for info
                            print(f"   ✓ Converted {name}: shape={shape}, sparsity={sparsity*100:.1f}%")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not convert {name} from sparse to dense: {e}")
                    # If conversion fails, try to keep the original (might be dense already)
                    dense_state_dict[name] = param
            else:
                # Already dense or not a tensor, keep as-is
                dense_state_dict[name] = param
        
        print(f"   Converted {converted_count} sparse tensors to dense format")
        
        # Create new checkpoint with dense state dict
        new_checkpoint = {
            'state_dict': dense_state_dict,
            'metadata': {
                **metadata,
                'sparse_format': False,  # Mark as converted
                'was_sparse': True,  # Remember it was originally sparse
                'conversion_info': {
                    'converted_tensors': converted_count,
                    'original_format': metadata.get('format', 'unknown')
                }
            }
        }
        
        # Preserve any other keys from original checkpoint
        for key, value in checkpoint.items():
            if key not in ['state_dict', 'metadata']:
                new_checkpoint[key] = value
        
        print("   ✓ Sparse checkpoint successfully converted to dense format")
        return new_checkpoint

