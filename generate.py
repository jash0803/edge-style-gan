import argparse
import os
import cv2
import torch
import numpy as np
import warnings
from core.utils import load_cfg, load_weights, tensor_to_img
from core.distiller import Distiller
from core.model_zoo import model_zoo
from core.compression import ModelCompressor
from tqdm import tqdm

# Suppress qnnpack reduce_range warning (it's a known issue, not critical)
# This warning appears when using qnnpack backend - it's harmless
warnings.filterwarnings('ignore', message='.*qnnpack incorrectly ignores reduce_range.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*reduce_range.*')

@torch.no_grad()
def main(args):
    cfg = load_cfg(args.cfg)
    distiller = Distiller(cfg)
    if args.ckpt is not None:
        # Set quantization engine BEFORE loading (model_zoo will handle this)
        # This prevents "Unknown qengine" error when loading quantized models
        try:
            available_engines = torch.backends.quantized.supported_engines
            # Default to qnnpack on Mac, fbgemm on Linux/Windows
            if 'qnnpack' in available_engines:
                torch.backends.quantized.engine = 'qnnpack'
            elif 'fbgemm' in available_engines:
                torch.backends.quantized.engine = 'fbgemm'
        except Exception:
            pass
        
        ckpt = model_zoo(args.ckpt)
        
        # Check if this is a quantized model and set quantization engine if needed
        if isinstance(ckpt, dict) and 'compression_method' in ckpt:
            compression_method = ckpt.get('compression_method', '')
            compression_config = ckpt.get('compression_config', {})
            
            if 'quantize' in compression_method.lower() or 'quantized' in compression_method.lower():
                # Set quantization engine for quantized models
                try:
                    # torch is already imported at the top, no need to re-import
                    # Check available engines
                    available_engines = torch.backends.quantized.supported_engines
                    if 'qnnpack' in available_engines:
                        torch.backends.quantized.engine = 'qnnpack'
                        print(f"✓ Set quantization engine: qnnpack")
                    elif 'fbgemm' in available_engines:
                        torch.backends.quantized.engine = 'fbgemm'
                        print(f"✓ Set quantization engine: fbgemm")
                    
                    # Get backend from config if available
                    backend = compression_config.get('backend', None)
                    if backend and backend in available_engines:
                        torch.backends.quantized.engine = backend
                        print(f"✓ Using quantization engine: {backend} (from config)")
                    
                    # Verify engine is set
                    current_engine = torch.backends.quantized.engine
                    print(f"✓ Active quantization engine: {current_engine}")
                    
                except Exception as e:
                    print(f"Warning: Could not set quantization engine: {e}")
                    print("Quantized model may not work correctly.")
        
        # Check if checkpoint contains only student model weights
        state_dict = ckpt.get("state_dict", {})
        
        # Check if this is a sparse checkpoint (has metadata with sparse_format)
        is_sparse_checkpoint = isinstance(ckpt, dict) and 'metadata' in ckpt and ckpt.get('metadata', {}).get('sparse_format', False)
        
        if is_sparse_checkpoint:
            print("📦 Sparse checkpoint detected. Converting to dense format for inference...")
            try:
                # Load and convert sparse checkpoint to dense
                ckpt = ModelCompressor.load_sparse_checkpoint(args.ckpt, map_location='cpu')
                print("✓ Sparse checkpoint successfully converted to dense format")
            except Exception as e:
                print(f"⚠️  Error converting sparse checkpoint: {e}")
                print("   Falling back to direct loading (may fail if sparse tensors present)...")
                # Try to continue anyway - might work if conversion partially succeeded
        
        # Check if this is a compressed checkpoint (has compression_method key)
        is_compressed = isinstance(ckpt, dict) and 'compression_method' in ckpt
        
        if is_compressed:
            compression_method = ckpt.get('compression_method', '')
            
            # Check if full quantized model was saved
            if 'model' in ckpt:
                # Full quantized model (GraphModule) was saved
                print("Loading quantized model (GraphModule)...")
                distiller.student = ckpt['model']
                print(f"Loaded {compression_method} quantized model")
                
                # Verify quantization by checking for quantized operations
                print("\n🔍 Verifying quantization status:")
                model = distiller.student
                quantized_ops = 0
                total_ops = 0
                quantized_modules = []
                
                # Check graph nodes for quantized operations
                if hasattr(model, 'graph'):
                    for node in model.graph.nodes:
                        total_ops += 1
                        node_str = str(node.target).lower()
                        if 'quantized' in node_str or 'quant' in node_str:
                            quantized_ops += 1
                
                # Check for quantized modules
                for name, module in model.named_modules():
                    if hasattr(module, 'qscheme') or 'quant' in str(type(module)).lower():
                        quantized_modules.append(name)
                
                # Check current quantization engine
                current_engine = torch.backends.quantized.engine
                
                if quantized_ops > 0 or len(quantized_modules) > 0:
                    print(f"  ✓ Found {quantized_ops} quantized operations in graph")
                    print(f"  ✓ Found {len(quantized_modules)} quantized modules")
                    if len(quantized_modules) > 0:
                        print(f"  ✓ Quantized modules (first 5): {quantized_modules[:5]}")
                    print(f"  ✓ Quantization engine: {current_engine}")
                    print(f"  ✓ Model is using quantized inference")
                else:
                    print(f"  ⚠️  Warning: No quantized operations detected")
                    print(f"  ✓ Quantization engine: {current_engine}")
                    print(f"  ℹ️  Model may be using quantized operations internally")
            elif 'quantize' in compression_method.lower():
                # Quantized model saved as state_dict
                # WARNING: Post-training quantization often breaks GAN models
                print("⚠️  WARNING: Quantized model detected.")
                print("   Post-training quantization of GANs often produces poor results.")
                print("   The model may generate colors/patterns instead of faces.")
                print("   Recommendation: Use a non-quantized checkpoint for best results.")
                print("   Loading weights anyway...")
                
                # Load weights directly without re-quantization
                if any(k.startswith("student.") for k in state_dict.keys()):
                    from core.utils import select_weights
                    student_state = select_weights(state_dict, prefix="student.")
                else:
                    student_state = state_dict
                
                distiller.student.load_state_dict(student_state, strict=False)
                print(f"   Loaded {len(student_state)} weight tensors")
                print("   Model running in float32 mode (quantization not active)")
            else:
                # Pruned model - just load state_dict
                if any(k.startswith("student.") for k in state_dict.keys()):
                    from core.utils import select_weights
                    student_state = select_weights(state_dict, prefix="student.")
                else:
                    student_state = state_dict
                
                distiller.student.load_state_dict(student_state, strict=False)
                print(f"Loaded {compression_method} student model weights")
                print(f"Loaded {len(student_state)} weight tensors")
        else:
            # Regular checkpoint - load all weights
            load_weights(distiller, state_dict)

    distiller = distiller.to(args.device)
    distiller.eval()  # Ensure model is in eval mode for inference
    
    # For quantized models, detect output range and adjust normalization
    is_quantized = isinstance(ckpt, dict) and 'compression_method' in ckpt and 'quantize' in ckpt.get('compression_method', '').lower()
    output_range = None
    
    if is_quantized:
        # Test a forward pass to detect actual output range
        print("Detecting quantized model output range...")
        with torch.no_grad():
            test_var = torch.randn(1, distiller.mapping_net.style_dim).to(args.device)
            test_img = distiller(test_var, truncated=args.truncated, generator=args.generator)
            img_min, img_max = test_img.min().item(), test_img.max().item()
            output_range = (img_min, img_max)
            print(f"Detected output range: [{img_min:.3f}, {img_max:.3f}]")
            
            # If range is very small, use adaptive normalization
            if abs(img_max - img_min) < 0.5:
                print("Output range is small - using adaptive normalization")
                # Use actual range with some padding
                output_range = (img_min - 0.1, img_max + 0.1)
            else:
                # Use standard range
                output_range = (-1.0, 1.0)
    
    for i in tqdm(range(args.n_batches)):
        var = torch.randn(args.batch_size, distiller.mapping_net.style_dim).to(args.device)
        img_s = distiller(var, truncated=args.truncated, generator=args.generator)
        for j in range(img_s.size(0)):
            # Use detected range for quantized models, otherwise use default
            if output_range:
                img_array = tensor_to_img(img_s[j].cpu(), normalize=True, range=output_range)
            else:
                img_array = tensor_to_img(img_s[j].cpu())
            cv2.imwrite(
                os.path.join(args.output_path, f"{i*args.batch_size + j}.png"),
                img_array
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--device", type=str, default="cpu", help="select device for inference")
    parser.add_argument("--cfg", type=str, default="configs/mobile_stylegan_ffhq.json", help="path to config file")
    parser.add_argument("--ckpt", type=str, default="mobilestylegan_ffhq.ckpt", help="path to checkpoint")
    parser.add_argument("--truncated", action='store_true', help="use truncation mode")
    parser.add_argument("--output-path", type=str, default="./", help="path to store images")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size")
    parser.add_argument("--n-batches", type=int, default=5000, help="number of batches")
    parser.add_argument("--generator", type=str, default="student", help="generator mode: [student|teacher]")
    args = parser.parse_args()
    main(args)
