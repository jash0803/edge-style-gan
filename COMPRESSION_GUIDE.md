# Model Compression Guide

This guide explains how to use quantization and pruning to further compress your MobileStyleGAN models.

## Overview

The compression module provides two main techniques:

1. **Pruning**: Removes less important weights from the model
   - **Unstructured pruning**: Removes individual weights (faster inference, no size reduction)
   - **Structured pruning**: Removes entire channels/filters (reduces file size)
   - **Sparse format**: Converts pruned models to sparse storage (reduces file size)

2. **Quantization**: Reduces precision of model weights
   - Static quantization: Post-training quantization (int8)
   - Dynamic quantization: Weights quantized, activations float
   - Quantization-Aware Training (QAT): Best accuracy, requires retraining

## Quick Start

### Basic Usage

```bash
# Compress with both pruning and quantization (recommended)
python compress_model.py \
    --cfg configs/mobile_stylegan_ffhq.json \
    --ckpt mobilestylegan_ffhq.ckpt \
    --method both \
    --amount 0.2 \
    --output mobilestylegan_compressed.ckpt
```

### Unstructured Pruning (Default)

Removes individual weights. Speeds up inference but doesn't reduce file size.

```bash
python compress_model.py \
    --cfg configs/mobile_stylegan_ffhq.json \
    --ckpt mobilestylegan_ffhq.ckpt \
    --method prune \
    --prune-type unstructured \
    --amount 0.3 \
    --prune-method magnitude \
    --output mobilestylegan_pruned.ckpt
```

### Structured Pruning

Removes entire channels/filters. **Actually reduces model file size**.

```bash
python compress_model.py \
    --cfg configs/mobile_stylegan_ffhq.json \
    --ckpt mobilestylegan_ffhq.ckpt \
    --method prune \
    --prune-type structured \
    --amount 0.3 \
    --prune-dim 0 \
    --output mobilestylegan_structured_pruned.ckpt
```

**Options:**
- `--prune-dim 0`: Prune output channels (default)
- `--prune-dim 1`: Prune input channels

### Unstructured Pruning + Sparse Format

Convert pruned model to sparse format to reduce file size:

```bash
python compress_model.py \
    --cfg configs/mobile_stylegan_ffhq.json \
    --ckpt mobilestylegan_ffhq.ckpt \
    --method prune \
    --prune-type unstructured \
    --amount 0.4 \
    --sparse-format coo \
    --output mobilestylegan_sparse.ckpt
```

This creates:
- `mobilestylegan_sparse.ckpt` - Regular checkpoint (for inference)
- `mobilestylegan_sparse_sparse.ckpt` - Sparse format (smaller file size)

**⚠️  IMPORTANT:** 
- **Use the regular checkpoint** (`mobilestylegan_sparse.ckpt`) **for inference/generating images**
- **DO NOT use the sparse checkpoint** (`mobilestylegan_sparse_sparse.ckpt`) for inference - it will produce corrupted/grey images
- The sparse checkpoint is **only for storage/archival purposes** to save disk space
- Sparse tensors cannot be used directly in model inference and will corrupt the weights

### Quantization Only

```bash
python compress_model.py \
    --cfg configs/mobile_stylegan_ffhq.json \
    --ckpt mobilestylegan_ffhq.ckpt \
    --method quantize \
    --backend fbgemm \
    --output mobilestylegan_quantized.ckpt
```

## Parameters

### Pruning Parameters

- `--amount`: Fraction of weights to prune (0.0 to 1.0)
  - `0.1` = 10% of weights removed
  - `0.2` = 20% of weights removed (recommended starting point)
  - `0.3` = 30% of weights removed
  - Higher values may significantly impact quality

- `--prune-method`: Pruning strategy
  - `magnitude`: Removes smallest magnitude weights (recommended)
  - `random`: Randomly removes weights (for comparison)

### Quantization Parameters

- `--backend`: Quantization backend
  - `fbgemm`: For CPU inference (recommended for desktop)
  - `qnnpack`: For mobile/ARM devices

## Using Compressed Models

After compression, use the compressed model just like the original:

```bash
# Generate images with compressed model
python generate.py \
    --cfg configs/mobile_stylegan_ffhq.json \
    --ckpt mobilestylegan_compressed.ckpt \
    --device cpu \
    --output-path ./outputs \
    --batch-size 5 \
    --n-batches 10
```

## Programmatic Usage

You can also use compression programmatically:

```python
from core.distiller import Distiller
from core.utils import load_cfg, load_weights
from core.model_zoo import model_zoo

# Load model
cfg = load_cfg("configs/mobile_stylegan_ffhq.json")
distiller = Distiller(cfg)
ckpt = model_zoo("mobilestylegan_ffhq.ckpt")
load_weights(distiller, ckpt["state_dict"])

# Apply compression
distiller.compress_model(
    compression_type='both',
    prune_amount=0.2,
    prune_method='magnitude',
    backend='fbgemm'
)

# Get model statistics
stats = distiller.get_model_stats()
print(f"Model size: {stats['size_mb']:.2f} MB")
print(f"Sparsity: {stats['sparsity']*100:.2f}%")
```

## Compression Strategies

### Conservative (Quality Priority)
- Pruning: 10-20%
- Quantization: Static quantization only
- Expected compression: 2-3x
- Quality impact: Minimal

### Balanced (Recommended)
- Pruning: 20-30%
- Quantization: Static quantization
- Expected compression: 3-5x
- Quality impact: Small, usually acceptable

### Aggressive (Size Priority)
- Pruning: 30-50%
- Quantization: Static quantization
- Expected compression: 5-10x
- Quality impact: Noticeable, may require retraining

## Iterative Pruning

For better results, you can use iterative pruning:

```python
from core.compression import ModelCompressor

compressor = ModelCompressor(model)

# Prune gradually: 10% -> 20% -> 30% -> 40%
compressor.prune_iterative(
    amounts=[0.1, 0.2, 0.3, 0.4],
    method='magnitude',
    retrain_fn=retrain_function  # Optional: retrain between steps
)
```

## Best Practices

1. **Start Small**: Begin with 10-20% pruning to assess quality impact
2. **Test Quality**: Always test compressed models on sample images
3. **Iterative Approach**: Use iterative pruning for better results
4. **Retrain After Pruning**: Consider fine-tuning after aggressive pruning
5. **Backend Selection**: Use `fbgemm` for CPU, `qnnpack` for mobile
6. **Save Original**: Keep original model for comparison

## Troubleshooting

### Quantization Errors
- If static quantization fails, the script will fall back to dynamic quantization
- Some operations may not support quantization (e.g., custom CUDA ops)
- Try quantization alone first, then add pruning

### Pruning Issues
- If model becomes too sparse, quality may degrade significantly
- Try lower pruning amounts or use iterative pruning
- Consider retraining after pruning

### Model Size Not Reduced
- Check if pruning was applied correctly (check sparsity)
- Quantization may not reduce file size if model is saved in float format
- Use `torch.save` with compressed format

## Technical Details

### Pruning Implementation
- Uses PyTorch's `torch.nn.utils.prune` module
- Global unstructured pruning across all layers
- L1 norm for magnitude-based pruning
- Pruning is made permanent (weights set to zero)

### Quantization Implementation
- Uses PyTorch's FX Graph Mode quantization
- Calibration with example inputs
- INT8 quantization for weights and activations
- Supports both CPU (fbgemm) and mobile (qnnpack) backends

## References

- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- MobileStyleGAN Paper: [arXiv:2104.04767](https://arxiv.org/abs/2104.04767)

