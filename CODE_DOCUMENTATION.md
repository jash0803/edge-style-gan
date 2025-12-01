# MobileStyleGAN.pytorch - Code Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [File Structure](#file-structure)
4. [Execution Flow](#execution-flow)
5. [Key Components](#key-components)
6. [Data Flow](#data-flow)

---

## Project Overview

MobileStyleGAN is a lightweight, compressed version of StyleGAN2 designed for high-fidelity image synthesis on mobile/edge devices. The project uses **knowledge distillation** to train a smaller student network (MobileStyleGAN) to mimic a larger teacher network (StyleGAN2).

### Key Features:
- **Knowledge Distillation**: Teacher-student training paradigm
- **DWT-based Architecture**: Uses Discrete Wavelet Transform for efficient upsampling
- **Mobile-Optimized**: Reduced model size while maintaining quality
- **Deployment Ready**: Supports ONNX, CoreML, and OpenVINO export

---

## Architecture Overview

### Teacher Model (StyleGAN2)
- **Mapping Network**: Transforms random noise `z` to style codes `w`
- **Synthesis Network**: Generates images from style codes using standard convolutions

### Student Model (MobileStyleGAN)
- **Mapping Network**: Same as teacher (shared)
- **Mobile Synthesis Network**: 
  - Uses **Depthwise Convolutions** instead of standard convolutions
  - Uses **DWT (Discrete Wavelet Transform)** for upsampling instead of bilinear interpolation
  - Operates in frequency domain for efficiency

---

## File Structure

### Root Level Scripts

#### `train.py`
**Purpose**: Main training script for knowledge distillation

**What it does**:
- Loads configuration from JSON
- Initializes Distiller (teacher + student models)
- Sets up PyTorch Lightning trainer
- Handles model export (ONNX, CoreML)

**Key Functions**:
- `build_logger()`: Creates logging interface (Neptune, TensorBoard, etc.)
- `main()`: Orchestrates training pipeline

**Execution Order**:
1. Parse command-line arguments
2. Load configuration file
3. Initialize Distiller
4. Load checkpoint (if provided)
5. Setup PyTorch Lightning trainer
6. Start training loop OR export model

---

#### `demo.py`
**Purpose**: Interactive demo for visualizing generated images

**What it does**:
- Loads trained model
- Generates random images in real-time
- Displays images using OpenCV
- Allows switching between student/teacher generators

**Execution Order**:
1. Load configuration
2. Initialize Distiller
3. Load checkpoint weights
4. **Loop**:
   - Generate random noise
   - Forward pass through model
   - Display image
   - Wait for user input (press 'q' to quit)

---

#### `generate.py`
**Purpose**: Batch image generation script

**What it does**:
- Generates multiple images in batches
- Saves images to disk as PNG files
- Supports batch processing for efficiency

**Execution Order**:
1. Load configuration
2. Initialize Distiller
3. Load checkpoint weights
4. Move model to specified device (CPU/CUDA)
5. **For each batch**:
   - Generate random noise
   - Forward pass
   - Save images to disk

---

#### `compare.py`
**Purpose**: Side-by-side comparison of teacher vs student outputs

**What it does**:
- Generates images from both teacher and student models
- Displays them side-by-side
- Interactive visualization

**Execution Order**:
1. Load configuration
2. Initialize Distiller
3. Load checkpoint weights
4. **Loop**:
   - Generate random noise
   - Forward pass through both models (`simultaneous_forward`)
   - Concatenate images horizontally
   - Display comparison
   - Wait for user input

---

#### `evaluate_fid.py`
**Purpose**: Calculate FID (Fréchet Inception Distance) score

**What it does**:
- Computes FID between real and generated images
- Uses InceptionV3 features
- Provides quantitative quality metric

**Execution Order**:
1. Load InceptionV3 model
2. Extract features from reference dataset
3. Extract features from generated images
4. Calculate FID score (statistical distance)

---

#### `convert_rosinality_ckpt.py`
**Purpose**: Convert StyleGAN2 checkpoints from rosinality format

**What it does**:
- Converts StyleGAN2 checkpoints to MobileStyleGAN format
- Separates mapping network and synthesis network
- Creates compatible checkpoint files

**Execution Order**:
1. Load rosinality checkpoint
2. Extract mapping network weights
3. Extract synthesis network weights
4. Save separate checkpoint files
5. Generate configuration JSON

---

### Core Module (`core/`)

#### `core/distiller.py`
**Purpose**: Main distillation class - orchestrates teacher-student training

**Key Components**:
- `Distiller` class (inherits from `pl.LightningModule`):
  - **Teacher Models**: Pre-trained StyleGAN2 mapping and synthesis networks
  - **Student Model**: MobileSynthesisNetwork (trainable)
  - **Loss Functions**: DistillerLoss (L1, L2, perceptual, GAN losses)
  - **Evaluation**: KID metric using InceptionV3

**Key Methods**:
- `__init__()`: Loads teacher models, initializes student
- `make_sample()`: Generates teacher outputs for distillation
- `generator_step()`: Trains student generator
- `discriminator_step()`: Trains discriminator
- `forward()`: Inference (student or teacher)
- `simultaneous_forward()`: Generate from both models
- `to_onnx()` / `to_coreml()`: Model export

**Execution Flow (Training)**:
1. Generate random noise `z`
2. Map to style codes `w` using teacher mapping network
3. Generate teacher image using teacher synthesis network
4. Generate student image using student synthesis network
5. Compute loss (L1, L2, perceptual, GAN)
6. Backpropagate and update student weights

---

#### `core/utils.py`
**Purpose**: Utility functions for model loading, configuration, and image processing

**Key Functions**:
- `download_ckpt()`: Downloads or loads local checkpoints
  - Checks current directory first
  - Falls back to `/tmp/` cache
  - Downloads from Google Drive if needed
- `load_cfg()`: Loads JSON configuration files
- `load_weights()`: Loads model weights with size matching
- `tensor_to_img()`: Converts PyTorch tensors to OpenCV images
- `select_weights()`: Filters weights by prefix
- `apply_trace_model_mode()`: Sets trace mode for ONNX export

---

#### `core/model_zoo.py`
**Purpose**: Model checkpoint management

**What it does**:
- Maps model names to download URLs
- Handles checkpoint loading from model zoo or local files
- Integrates with `download_ckpt()` utility

**Execution Flow**:
1. Load `model_zoo.json` configuration
2. Check if name exists in zoo
3. If yes: Download/load from zoo
4. If no: Load as local file path

---

#### `core/dataset.py`
**Purpose**: Dataset for training (noise generation)

**What it does**:
- Generates random noise vectors for training
- No actual image dataset needed (unsupervised)

**Key Class**:
- `NoiseDataset`: Generates random noise on-the-fly

---

### Loss Functions (`core/loss/`)

#### `core/loss/distiller_loss.py`
**Purpose**: Combined loss function for knowledge distillation

**Components**:
- `DistillerLoss` class:
  - **L1 Loss**: Pixel-level difference
  - **L2 Loss**: Mean squared error
  - **Perceptual Loss**: VGG16 feature matching
  - **GAN Loss**: Adversarial training
  - **DWT Loss**: Frequency domain matching

**Key Methods**:
- `loss_g()`: Generator loss (student vs teacher)
- `loss_d()`: Discriminator loss
- `reg_d()`: Discriminator regularization
- `img_to_dwt()` / `dwt_to_img()`: Domain conversion

**Loss Computation**:
1. Compute losses in spatial domain (L1, L2 on RGB images)
2. Compute losses in frequency domain (L1, L2 on DWT coefficients)
3. Compute perceptual loss (VGG16 features)
4. Compute GAN loss (discriminator output)
5. Weighted sum of all losses

---

#### `core/loss/perceptual_loss.py`
**Purpose**: Perceptual loss using VGG16 features

**Components**:
- `PerceptualNetwork`: Extracts VGG16 features
- `PerceptualLoss`: Computes L1 loss on feature maps

**What it does**:
- Extracts features from multiple VGG16 layers
- Compares features between student and teacher outputs
- Provides semantic similarity measure

---

#### `core/loss/non_saturating_gan_loss.py`
**Purpose**: Non-saturating GAN loss implementation

**What it does**:
- Implements standard GAN training loss
- Supports discriminator regularization

---

#### `core/loss/diffaug.py`
**Purpose**: Differential augmentation for training stability

---

### Models (`core/models/`)

#### `core/models/mapping_network.py`
**Purpose**: StyleGAN mapping network (shared by teacher and student)

**Architecture**:
- Multi-layer MLP
- Transforms `z` (latent code) → `w` (style code)

---

#### `core/models/synthesis_network.py`
**Purpose**: StyleGAN2 synthesis network (teacher model)

**Architecture**:
- Standard StyleGAN2 synthesis blocks
- Uses regular convolutions
- Bilinear upsampling

---

#### `core/models/mobile_synthesis_network.py`
**Purpose**: Mobile-optimized synthesis network (student model)

**Key Differences from Teacher**:
- Uses **DWT upsampling** instead of bilinear
- Uses **depthwise convolutions** for efficiency
- Operates in frequency domain

**Architecture**:
- `ConstantInput`: Initial constant tensor
- `StyledConv2d`: Style-modulated convolution
- `MobileSynthesisBlock`: Main building blocks
- `DWTInverse`: Converts frequency domain to RGB

**Forward Pass**:
1. Start with constant input
2. Apply initial styled convolution
3. Convert to frequency domain (12 channels: 3 low + 9 high)
4. **For each block**:
   - Upsample using DWT
   - Apply styled convolutions
   - Generate frequency representation
5. Final DWT inverse to RGB image

---

#### `core/models/inception_v3.py`
**Purpose**: InceptionV3 for FID/KID evaluation

**Components**:
- `InceptionV3`: Feature extractor
- `load_inception_v3()`: Loads pretrained model
- `fid_inception_v3()`: FID-specific Inception model

---

### Model Modules (`core/models/modules/`)

#### `core/models/modules/mobile_synthesis_block.py`
**Purpose**: Building block for mobile synthesis network

**Components**:
- `IDWTUpsaplme`: DWT-based upsampling
- `StyledConv2d`: Style-modulated convolutions (x2)
- `MultichannelIamge`: Converts to frequency domain

**Forward Pass**:
1. Upsample using DWT
2. First styled convolution
3. Second styled convolution
4. Convert to frequency representation

---

#### `core/models/modules/idwt_upsample.py`
**Purpose**: DWT-based upsampling module

**What it does**:
- Uses Inverse Discrete Wavelet Transform for upsampling
- More efficient than bilinear interpolation
- Maintains frequency information

---

#### `core/models/modules/idwt.py`
**Purpose**: Inverse DWT implementation

**What it does**:
- Reconstructs images from frequency domain
- Converts DWT coefficients back to RGB

---

#### `core/models/modules/modulated_conv2d.py`
**Purpose**: Style-modulated convolution (standard)

---

#### `core/models/modules/styled_conv2d.py`
**Purpose**: Wrapper for style-modulated convolutions

**What it does**:
- Applies style modulation
- Handles noise injection
- Supports both standard and depthwise convolutions

---

#### `core/models/modules/multichannel_image.py`
**Purpose**: Converts feature maps to frequency domain

**What it does**:
- Transforms feature channels to DWT representation
- Outputs 12 channels (3 low-frequency + 9 high-frequency)

---

#### `core/models/modules/noise_injection.py`
**Purpose**: Adds noise to feature maps

---

#### `core/models/modules/constant_input.py`
**Purpose**: Initial constant input tensor

---

#### `core/models/modules/ops/`
**Purpose**: Custom CUDA operations

**Files**:
- `fused_act.py` / `fused_act_cuda.py`: Fused activation functions
- `upfirdn2d.py` / `upfirdn2d_cuda.py`: Upsampling/filtering operations
- CUDA kernels for performance optimization

---

### Configuration Files (`configs/`)

#### `configs/mobile_stylegan_ffhq.json`
**Purpose**: Main training configuration

**Sections**:
- `logger`: Logging configuration (Neptune)
- `trainset` / `valset`: Dataset parameters
- `teacher`: Teacher model checkpoint paths
- `distillation_loss`: Loss weights
- `trainer`: Training hyperparameters

---

#### `configs/model_zoo.json`
**Purpose**: Model checkpoint registry

**What it contains**:
- Model names → Download URLs
- MD5 checksums for verification

---

## Execution Flow

### Training Flow (`train.py`)

```
1. Parse arguments
   ↓
2. Load config (mobile_stylegan_ffhq.json)
   ↓
3. Initialize Distiller
   ├─ Load teacher mapping network (from model_zoo)
   ├─ Load teacher synthesis network (from model_zoo)
   ├─ Initialize student MobileSynthesisNetwork
   ├─ Setup DistillerLoss
   └─ Setup InceptionV3 for evaluation
   ↓
4. Load checkpoint (if provided)
   ↓
5. Setup PyTorch Lightning Trainer
   ↓
6. Start training loop
   ├─ For each batch:
   │  ├─ Generate random noise z
   │  ├─ Map to style w (teacher mapping)
   │  ├─ Generate teacher image
   │  ├─ Generate student image
   │  ├─ Compute losses
   │  └─ Update student weights
   └─ Validation (compute KID)
```

### Inference Flow (`demo.py`, `generate.py`)

```
1. Load config
   ↓
2. Initialize Distiller
   ├─ Load teacher models
   └─ Initialize student model
   ↓
3. Load checkpoint weights
   ↓
4. Generate images
   ├─ Random noise z
   ├─ Map to style w
   └─ Forward through generator (student/teacher)
   ↓
5. Display/Save images
```

### Knowledge Distillation Process

```
Training Step:
┌─────────────────┐
│ Random Noise z  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mapping Network │ (Teacher - frozen)
└────────┬────────┘
         │
         ▼
    ┌────┴────┐
    │        │
    ▼        ▼
┌──────────┐  ┌──────────────────┐
│ Teacher  │  │ Student          │
│ Synthesis│  │ Mobile Synthesis │
└────┬─────┘  └────────┬─────────┘
     │                  │
     │                  │
     ▼                  ▼
┌──────────┐      ┌──────────┐
│ Teacher  │      │ Student  │
│ Image    │      │ Image    │
└────┬─────┘      └────┬─────┘
     │                 │
     └────────┬────────┘
              │
              ▼
      ┌───────────────┐
      │ DistillerLoss │
      │ - L1/L2       │
      │ - Perceptual  │
      │ - GAN         │
      │ - DWT         │
      └───────┬───────┘
              │
              ▼
      ┌───────────────┐
      │ Backprop      │
      │ Update Student│
      └───────────────┘
```

---

## Key Components

### 1. Knowledge Distillation
- **Teacher**: Pre-trained StyleGAN2 (frozen)
- **Student**: MobileStyleGAN (trainable)
- **Process**: Student learns to mimic teacher outputs

### 2. DWT (Discrete Wavelet Transform)
- **Purpose**: Efficient upsampling and frequency domain processing
- **Usage**: 
  - Upsampling in mobile synthesis blocks
  - Loss computation in frequency domain
  - Final image reconstruction

### 3. Depthwise Convolutions
- **Purpose**: Reduce model parameters
- **Usage**: Replaces standard convolutions in student model

### 4. Style Modulation
- **Purpose**: Control image generation via style codes
- **Usage**: Applied in all synthesis blocks

### 5. Multi-Scale Loss
- **L1/L2**: Pixel-level matching
- **Perceptual**: Feature-level matching (VGG16)
- **GAN**: Adversarial training
- **DWT**: Frequency domain matching

---

## Data Flow

### Image Generation Pipeline

```
Random Noise (z)
    │
    ▼
Mapping Network
    │
    ▼
Style Codes (w)
    │
    ├─────────────────┐
    │                 │
    ▼                 ▼
Teacher          Student
Synthesis       Mobile Synthesis
    │                 │
    │                 │
    │         ┌───────┴───────┐
    │         │               │
    │         ▼               ▼
    │    Frequency      Frequency
    │    Domain         Domain
    │         │               │
    │         └───────┬───────┘
    │                 │
    ▼                 ▼
RGB Image        RGB Image
(Teacher)        (Student)
```

### Training Data Flow

```
NoiseDataset
    │
    ▼
Random z vectors
    │
    ▼
Distiller.make_sample()
    │
    ├─→ Teacher Mapping → Teacher Synthesis → Teacher Image
    │
    └─→ Student Mobile Synthesis → Student Image
    │
    ▼
DistillerLoss
    │
    ├─→ L1/L2 Loss (spatial)
    ├─→ L1/L2 Loss (frequency)
    ├─→ Perceptual Loss (VGG16)
    └─→ GAN Loss
    │
    ▼
Total Loss
    │
    ▼
Backpropagation
    │
    ▼
Update Student Weights
```

---

## File Dependencies

### Core Dependencies
```
train.py / demo.py / generate.py
    │
    ├─→ core.distiller
    │       ├─→ core.models.mapping_network
    │       ├─→ core.models.synthesis_network (teacher)
    │       ├─→ core.models.mobile_synthesis_network (student)
    │       ├─→ core.loss.distiller_loss
    │       └─→ core.models.inception_v3
    │
    ├─→ core.utils
    │       └─→ core.model_zoo
    │
    └─→ configs/*.json
```

### Model Dependencies
```
mobile_synthesis_network.py
    │
    ├─→ modules.mobile_synthesis_block
    │       ├─→ modules.idwt_upsample
    │       ├─→ modules.styled_conv2d
    │       └─→ modules.multichannel_image
    │
    ├─→ modules.constant_input
    ├─→ modules.idwt
    └─→ modules.modulated_conv2d
```

---

## Summary

This codebase implements **MobileStyleGAN**, a compressed version of StyleGAN2 using:

1. **Knowledge Distillation**: Teacher (StyleGAN2) → Student (MobileStyleGAN)
2. **DWT-based Architecture**: Efficient frequency domain processing
3. **Mobile Optimizations**: Depthwise convolutions, reduced parameters
4. **Multi-scale Training**: Spatial + frequency + perceptual losses

The main entry points are:
- **`train.py`**: Training the student model
- **`demo.py`**: Interactive visualization
- **`generate.py`**: Batch image generation
- **`compare.py`**: Teacher vs student comparison

All scripts follow a similar pattern: load config → initialize Distiller → load weights → generate/display images.

