import argparse
import os
import sys
import torch
from core.utils import load_cfg, load_weights
from core.distiller import Distiller
from core.model_zoo import model_zoo
from core.compression import ModelCompressor


def main(args):
    print("=" * 60)
    print("MobileStyleGAN Model Compression Tool")
    print("=" * 60)
    
    # Load configuration and model
    print(f"\n1. Loading configuration from {args.cfg}...")
    cfg = load_cfg(args.cfg)
    distiller = Distiller(cfg)
    
    # Load trained weights
    if args.ckpt:
        print(f"2. Loading checkpoint from {args.ckpt}...")
        ckpt = model_zoo(args.ckpt)
        load_weights(distiller, ckpt["state_dict"])
        print("   Checkpoint loaded successfully!")
    else:
        print("2. No checkpoint provided, using randomly initialized model")
    
    # Get student model
    student_model = distiller.student
    student_model.eval()
    
    # Initialize compressor
    compressor = ModelCompressor(student_model)
    
    # Get original model statistics
    original_size = compressor.get_model_size()
    total_params, trainable_params = compressor.get_num_parameters()
    
    print(f"\n3. Original Model Statistics:")
    print(f"   - Size: {original_size:.2f} MB")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Apply compression
    print(f"\n4. Applying compression: {args.method}")
    print("-" * 60)
    
    if args.method == 'quantize':
        print("   Method: Static Quantization")
        
        # Check available backends and adjust if needed
        try:
            available_engines = torch.backends.quantized.supported_engines
            print(f"   Available engines: {available_engines}")
            if args.backend not in available_engines:
                if 'qnnpack' in available_engines:
                    print(f"   Warning: {args.backend} not available. Auto-selecting qnnpack.")
                    args.backend = 'qnnpack'
                elif 'fbgemm' in available_engines:
                    print(f"   Warning: {args.backend} not available. Auto-selecting fbgemm.")
                    args.backend = 'fbgemm'
        except Exception:
            pass
        
        print(f"   Backend: {args.backend}")
        print("   Note: On Mac, use qnnpack. On Linux/Windows, use fbgemm.")
        
        # Prepare example inputs
        example_style = torch.randn(1, distiller.mapping_net.style_dim)
        example_w = distiller.mapping_net(example_style)
        example_inputs = (example_w,)
        
        try:
            quantized_model = compressor.quantize_static(
                example_inputs,
                backend=args.backend
            )
            distiller.student = quantized_model
            
            quantized_size = compressor.get_model_size(quantized_model)
            compression_ratio = original_size / quantized_size
            
            print(f"\n   Quantization Results:")
            print(f"   - Quantized size: {quantized_size:.2f} MB")
            print(f"   - Compression ratio: {compression_ratio:.2f}x")
            
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n   ⚠️  Quantization not available")
            if "Python 3.14" in error_msg or "compatibility" in error_msg.lower():
                print("   This is due to Python 3.14 compatibility issues with PyTorch quantization.")
                print("   Recommendation: Use pruning instead, or use Python 3.12 or earlier for quantization.")
            else:
                print(f"   Error: {error_msg}")
            print("\n   Please use --method prune instead, or skip quantization.")
            print("   Example: python compress_model.py --method prune --amount 0.2 ...")
            sys.exit(1)  # Exit gracefully with error code
        except Exception as e:
            print(f"   Error during quantization: {e}")
            print("   Attempting dynamic quantization as fallback...")
            try:
                quantized_model = compressor.quantize_dynamic()
                distiller.student = quantized_model
                quantized_size = compressor.get_model_size(quantized_model)
                compression_ratio = original_size / quantized_size
                print(f"   - Quantized size: {quantized_size:.2f} MB")
                print(f"   - Compression ratio: {compression_ratio:.2f}x")
            except Exception as e2:
                print(f"   Dynamic quantization also failed: {e2}")
                print("   Quantization is not available. Please use pruning instead.")
                print("   Example: python compress_model.py --method prune --amount 0.2 ...")
                sys.exit(1)  # Exit gracefully with error code
        
    elif args.method == 'prune':
        if args.prune_type == 'structured':
            print(f"   Method: Structured Pruning")
            print(f"   Amount: {args.amount*100:.1f}%")
            print(f"   ⚠️  Note: PyTorch structured pruning zeros channels but doesn't remove them.")
            print(f"      Model file size will NOT be reduced.")
            print(f"      For size reduction, use: unstructured pruning + --sparse-format")
            
            pruned_model = compressor.prune_structured(
                amount=args.amount,
                dim=args.prune_dim
            )
            distiller.student = pruned_model
            
            pruned_size = compressor.get_model_size()
            total_params, _ = compressor.get_num_parameters()
            sparsity = compressor.get_sparsity()
            compression_ratio = original_size / pruned_size
            
            print(f"\n   Structured Pruning Results:")
            print(f"   - Pruned size: {pruned_size:.2f} MB (same as original - channels zeroed, not removed)")
            print(f"   - Total parameters: {total_params:,} (unchanged)")
            print(f"   - Model sparsity: {sparsity*100:.2f}% (channels zeroed)")
            print(f"   - Compression ratio: {compression_ratio:.2f}x (no file size reduction)")
            print(f"   - Inference speed: Faster (zero channels can be skipped)")
            print(f"\n   💡 For actual size reduction, use:")
            print(f"      --prune-type unstructured --sparse-format coo")
        else:
            print(f"   Method: Unstructured Pruning")
            print(f"   Amount: {args.amount*100:.1f}%")
            print(f"   Pruning method: {args.prune_method}")
            print(f"   Note: Unstructured pruning doesn't reduce file size, but speeds up inference")
            
            pruned_model = compressor.prune_unstructured(
                amount=args.amount,
                method=args.prune_method
            )
            distiller.student = pruned_model
            
            pruned_size = compressor.get_model_size()
            sparsity = compressor.get_sparsity()
            compression_ratio = original_size / pruned_size
            
            print(f"\n   Unstructured Pruning Results:")
            print(f"   - Pruned size: {pruned_size:.2f} MB")
            print(f"   - Model sparsity: {sparsity*100:.2f}%")
            print(f"   - Compression ratio: {compression_ratio:.2f}x")
            
            # Optionally convert to sparse format
            if args.sparse_format:
                print(f"\n   Converting to sparse format ({args.sparse_format})...")
                print("   💡 Note: Sparse checkpoints are automatically converted to dense when loaded.")
                print("      They save disk space and can be used for inference (conversion is automatic).")
                compressor.convert_to_sparse(format=args.sparse_format)
                
                # Save sparse checkpoint
                sparse_path = args.output.replace('.ckpt', '_sparse.ckpt')
                sparse_size, dense_size = compressor.save_sparse_checkpoint(sparse_path, format=args.sparse_format)
                print(f"   Sparse checkpoint saved: {sparse_path}")
                print(f"   - Dense size: {dense_size:.2f} MB")
                print(f"   - Sparse size: {sparse_size:.2f} MB")
                if dense_size > 0:
                    print(f"   - Size reduction: {(1 - sparse_size/dense_size)*100:.1f}%")
                print(f"\n   ✅ Use '{args.output}' (regular checkpoint) for inference")
                print(f"   📦 Use '{sparse_path}' (sparse checkpoint) for storage - can also be used for inference")
                print(f"      (will be automatically converted to dense when loaded)")
        
    elif args.method == 'both':
        print(f"   Method: Pruning + Quantization")
        print(f"   Pruning amount: {args.amount*100:.1f}%")
        print(f"   Pruning method: {args.prune_method}")
        print(f"   Quantization backend: {args.backend}")
        print("   Note: If quantization fails, the pruned model will be saved instead.")
        
        # Step 1: Prune
        print(f"\n   Step 1: Pruning...")
        pruned_model = compressor.prune_unstructured(
            amount=args.amount,
            method=args.prune_method
        )
        distiller.student = pruned_model
        
        pruned_size = compressor.get_model_size()
        sparsity = compressor.get_sparsity()
        
        print(f"   - After pruning:")
        print(f"     Size: {pruned_size:.2f} MB")
        print(f"     Sparsity: {sparsity*100:.2f}%")
        
        # Step 2: Quantize
        print(f"\n   Step 2: Quantization...")
        example_style = torch.randn(1, distiller.mapping_net.style_dim)
        example_w = distiller.mapping_net(example_style)
        example_inputs = (example_w,)
        
        try:
            quantized_model = compressor.quantize_static(
                example_inputs,
                backend=args.backend
            )
            distiller.student = quantized_model
            
            final_size = compressor.get_model_size(quantized_model)
            final_compression_ratio = original_size / final_size
            
            print(f"   - After quantization:")
            print(f"     Final size: {final_size:.2f} MB")
            print(f"     Final compression ratio: {final_compression_ratio:.2f}x")
            
        except RuntimeError as e:
            print(f"\n   ⚠️  Quantization not available: {e}")
            print("   This is likely due to Python 3.14 compatibility issues with PyTorch quantization.")
            print("   Using pruned model only (quantization skipped).")
            print("   For quantization, please use Python 3.12 or earlier.")
            final_size = pruned_size
            final_compression_ratio = original_size / final_size
            print(f"   - Final size (pruned only): {final_size:.2f} MB")
            print(f"   - Compression ratio: {final_compression_ratio:.2f}x")
        except Exception as e:
            print(f"   Error during quantization: {e}")
            print("   Attempting dynamic quantization as fallback...")
            try:
                quantized_model = compressor.quantize_dynamic()
                distiller.student = quantized_model
                final_size = compressor.get_model_size(quantized_model)
                final_compression_ratio = original_size / final_size
                print(f"   - Final size: {final_size:.2f} MB")
                print(f"   - Compression ratio: {final_compression_ratio:.2f}x")
            except Exception as e2:
                print(f"   Dynamic quantization also failed: {e2}")
                print("   Using pruned model only (no quantization)")
                final_size = pruned_size
                final_compression_ratio = original_size / final_size
    
    # Save compressed model
    if args.output:
        print(f"\n5. Saving compressed model to {args.output}...")
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        
        # Check if model is quantized (GraphModule)
        is_quantized = hasattr(distiller.student, '_modules') and any(
            'quantized' in str(type(m)).lower() or 'graphmodule' in str(type(m)).lower()
            for m in distiller.student.modules()
        ) or 'GraphModule' in str(type(distiller.student))
        
        if is_quantized:
            # For quantized models, save the entire model (not just state_dict)
            # because quantized models are GraphModules with special structure
            save_dict = {
                'model': distiller.student,  # Save entire quantized model
                'state_dict': distiller.student.state_dict(),  # Also save state_dict as backup
                'compression_method': args.method,
                'compression_config': {
                    'amount': args.amount if args.method in ['prune', 'both'] else None,
                    'prune_method': args.prune_method if args.method in ['prune', 'both'] else None,
                    'backend': args.backend if args.method in ['quantize', 'both'] else None,
                },
                'model_stats': distiller.get_model_stats(),
                'config': cfg
            }
            print("   Saving quantized model (GraphModule) - full model saved")
        else:
            # For pruned models, just save state_dict
            save_dict = {
                'state_dict': distiller.student.state_dict(),
                'compression_method': args.method,
                'compression_config': {
                    'amount': args.amount if args.method in ['prune', 'both'] else None,
                    'prune_method': args.prune_method if args.method in ['prune', 'both'] else None,
                    'backend': args.backend if args.method in ['quantize', 'both'] else None,
                },
                'model_stats': distiller.get_model_stats(),
                'config': cfg
            }
        
        torch.save(save_dict, args.output)
        print(f"   Compressed model saved successfully!")
        
        # Print final summary
        print(f"\n" + "=" * 60)
        print("Compression Summary:")
        print("=" * 60)
        stats = distiller.get_model_stats()
        print(f"Original size:     {original_size:.2f} MB")
        print(f"Compressed size:   {stats['size_mb']:.2f} MB")
        print(f"Compression ratio: {original_size/stats['size_mb']:.2f}x")
        if stats['sparsity'] > 0:
            print(f"Model sparsity:     {stats['sparsity']*100:.2f}%")
        print(f"Total parameters:  {stats['total_params']:,}")
        print("=" * 60)
    else:
        print("\n5. No output path specified. Model compressed but not saved.")
        print("   Use --output to save the compressed model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress MobileStyleGAN model using quantization and/or pruning"
    )
    parser.add_argument(
        "--cfg", 
        type=str, 
        required=True, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True, 
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=['quantize', 'prune', 'both'], 
        default='both',
        help="Compression method: 'quantize', 'prune', or 'both'"
    )
    parser.add_argument(
        "--amount", 
        type=float, 
        default=0.2, 
        help="Pruning amount (0.0 to 1.0), default: 0.2"
    )
    parser.add_argument(
        "--prune-type",
        type=str,
        choices=['unstructured', 'structured'],
        default='unstructured',
        help="Pruning type: 'unstructured' (faster inference) or 'structured' (reduces size), default: 'unstructured'"
    )
    parser.add_argument(
        "--prune-method", 
        type=str, 
        choices=['magnitude', 'random'],
        default='magnitude',
        help="Pruning method for unstructured pruning: 'magnitude' or 'random', default: 'magnitude'"
    )
    parser.add_argument(
        "--prune-dim",
        type=int,
        default=0,
        help="Dimension to prune for structured pruning (0=output channels, 1=input channels), default: 0"
    )
    parser.add_argument(
        "--sparse-format",
        type=str,
        choices=['coo', 'csr', 'csc', None],
        default=None,
        help="Convert to sparse format after unstructured pruning: 'coo', 'csr', 'csc', or None (disabled), default: None"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=['fbgemm', 'qnnpack'],
        default='fbgemm',
        help="Quantization backend: 'fbgemm' (CPU) or 'qnnpack' (mobile), default: 'fbgemm'"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Output path for compressed model"
    )
    
    args = parser.parse_args()
    main(args)

