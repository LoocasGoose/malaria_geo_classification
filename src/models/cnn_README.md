# CNN Model Documentation

--------------------------------
## Overview

### Standard vs Advanced CNN Implementation

**cnn_standard.py** was the *first final version* of the model. It works, but I wanted to try out new features. Some are complicated (to me), some are just ideas I had that I want to play around with. So I made a copy, now **cnn_advanced.py**, and experimented. 

### Key Enhancements in Advanced Version (Standard ➞ Advanced)

| Feature | Standard Version | Advanced Version | Benefits/Details |
|---------|-----------------|------------------|------------------|
| Skip Connections | `ResidualStrandConv` | `DenseResidualBlock` | Used concepts from **ResNet**, **DenseNet**, and **ResNeXt** with **cardinality** for better gradient flow and feature reuse |
| Attention Mechanism | Single layer `AttentionPooling` | `HierarchicalAttention` | Apply attention to multiple layers |
| Positional Encoding | Simple encoding (max_len=2000) | Chromosome-aware `PositionalEncoding` | Added n_chromosomes=14 and larger context window (max_len=10000) |
| Variant Simulation | None | `simulate_variants` function | Models SNPs, insertions, and deletions |
| Sequence Augmentation | Simple reverse complement | Multiple strategies | `augment_sequence` (general mutation), `augment_sequence_with_rc` (improved RC handling), `add_positional_noise` (simulates sequencing errors) |
| Strand Information Fusion | Simple averaging weights `StrandSymmetricConv` | Learnable weights | Better detection of strand-biased patterns |

### Additional Improvements
- Mixed precision training
- Checkpoint management 
- Dynamic batch sizing based on available hardware
- Customizable configurable channels, kernel sizes, and FC layers

--------------------------------
## Model Evolution

As I was building the CNN model, I documented each major version below. Feel free to check them out, compare the differences between models, and explore my thought process!

### Version 1: Basic Functioning Model
- 32 filters detecting 5bp patterns, basic padding to maintain length
- Basic SGD optimizer, CrossEntropy loss, fixed learning rate
- Fixed sequence windows (no sliding)
- Raw logits for country probabilities
- Simple argmax prediction
- Single validation check per epoch

```
[Input: DNA Seq]
    ↓ (1000bp, 5 channels)
[Conv1D: 32 filters, 5bp kernel] → ReLU
    ↓
[MaxPool1D: stride 2]
    ↓ (500bp, 32 channels)
[Flatten]
    ↓ (16,000 features)
[Dense: 128 units] → ReLU
    ↓
[Dense: n_countries]
    ↓
[Softmax]
```

### Version 2: Depth & Basic Training, BIOS Optimization
- Add 2 more conv layers (64, 128 filters)
- Implement max pooling
- Add validation split
- Proper one-hot encoding (A/C/G/T/N)
- Sequence length standardization
- Basic data loader

```
[Input: DNA Seq]
    ↓ (1000bp, 5 channels)
[Conv1D: 32 filters, 5bp kernel] → ReLU
    ↓
[MaxPool1D: stride 2]
    ↓ (500bp, 32 channels)
[Conv1D: 64 filters, 5bp kernel] → ReLU
    ↓
[MaxPool1D: stride 2]
    ↓ (250bp, 64 channels)
[Conv1D: 128 filters, 5bp kernel] → ReLU
    ↓
[MaxPool1D: stride 2]
    ↓ (125bp, 128 channels)
[Flatten]
    ↓ (16,000 features)
[Dense: 128 units] → ReLU
    ↓
[Dense: n_countries]
    ↓
[Softmax]
```

### Version 3: Biological Awareness, Training Improvements
- Reverse complement augmentation
- Strand-symmetric conv base
- Simple attention mechanism
- Adam optimizer
- Learning rate scheduling
- Early stopping

```
[Input: DNA Seq]
    ↓ (1000bp, 5 channels)
           ↓                     ↓
   [Original Seq]        [Reverse Complement]
           ↓                     ↓
           → [Strand-Symmetric Conv1D: 32] →
                           ↓
                        [ReLU]
                           ↓
                     [MaxPool1D: 2]
                           ↓
           → [Strand-Symmetric Conv1D: 64] →
                           ↓
                        [ReLU]
                           ↓
                     [MaxPool1D: 2]
                           ↓
           → [Strand-Symmetric Conv1D: 128] →
                           ↓
                        [ReLU]
                           ↓
                    [Attention Layer]
                           ↓
                       [Flatten]
                           ↓
                   [Dense: 128] → ReLU
                           ↓
                   [Dense: n_countries]
                           ↓
                       [Softmax]
```

### Version 4 (this version is cnn_standard.py): Advanced Architecture and Regularization
- Residual connections 
- Batch normalization
- Positional encoding
- Dropout layers
- Label smoothing
- Gradient clipping

```
[Input: DNA Seq (1000bp, 5 channels)]
           ↓
[Positional Encoding] (64 positional channels)
           ↓
[Combined Features: 5+64 channels] 
           ︱←───────────────┐
           ↓                 │
    ┌───────────────┐        │
    │ SymConv1D(32) │        │
    │ BatchNorm     │        │
    │ ReLU          │        │
    └───────────────┘        │
           ︱←───────────────┐
           ↓                 │
    ┌───────────────┐        │
    │ SymConv1D(32) │        │
    │ BatchNorm     │        │
    │ Dropout(0.4)  │        │
    └───────────────┘        │
           ↓                 │
[Residual Connection] ───────┘
           ↓
           ︱←───────────────┐
           ↓                 │
    ┌───────────────┐        │
    │ SymConv1D(64) │        │
    │ BatchNorm     │        │
    │ ReLU          │        │
    └───────────────┘        │
           ︱←───────────────┐
           ↓                 │
    ┌───────────────┐        │
    │ SymConv1D(64) │        │
    │ BatchNorm     │        │
    │ Dropout(0.4)  │        │
    └───────────────┘        │
           ↓                 │
[Residual Connection] ───────┘
           ↓
[Attention Pooling]
           ↓
[Dense: 512] → ReLU → Dropout(0.4)
           ↓
[Dense: n_countries + Label Smoothing]
           ↓
       [Softmax]
```

### Version 5 (this version is cnn_advanced.py): Further Performance Tuning
Key improvements:
- Model checkpointing
- Metric tracking
- Interpretability
- Mixed precision training
- Memory optimizations
- Multi-GPU support
- Chromosome position handling
- Variant simulation
- Hierarchical attention mechanism
- Advanced region visualization
- Complex skip connections
- Learnable weights

```
                      [Input: DNA Seq]
                              ↓
    ┌────────────────────────┼────────────────────────┐
    ↓                         ↓                       ↓
[Original]              [RC Augmented]           [Variant Sim]
    ↓                         ↓                       ↓
    └────────────────────────┼────────────────────────┘
                              ↓
                    [Positional Encoding]
                              ↓
                  [Multi-GPU Distribution]
                              ↓
              [Mixed Precision Forward Pass]
   ┌───────────────────────────────────────────────┐
   │ ┌─────────────┐    ┌─────────────┐            │
   │ │ResidualBlock│→→→→│ResidualBlock│→→→         │
   │ │  Conv(32)   │    │  Conv(64)   │   │        │
   │ └─────────────┘    └─────────────┘   │        │
   │        ↑                             ↓        │
   │        └──────────────────[ResidualBlock]     │
   │                           │  Conv(128)  │     │
   │                           └─────────────┘     │
   └───────────────────────────────────────────────┘
                              ↓
                    [Hierarchical Attention]
                              ↓
                     [Region Visualization]
                              ↓
           [Checkpoint Manager + Metrics Tracking]
                              ↓
                    [Output Interpretation]
                              ↓
                       [Multi-class Pred]
```

--------------------------------
## Core Functionality

This custom CNN is specifically designed to analyze **Plasmodium falciparum** DNA sequences and predict their geographic origins. Unlike generic models, it processes DNA's double-stranded nature simultaneously using **strand-symmetric convolutions**, ensuring it detects patterns regardless of which DNA strand they appear on.

### Unique Features
The model incorporates **positional encoding** to understand genomic context and **attention mechanisms** to focus on biologically significant regions. Specialized 15bp, 9bp, and 5bp convolutional filters detect malaria-specific motifs like drug resistance markers and surface proteins, while **residual connections** enable stable training on deep networks. Built-in **data augmentation** simulates real-world sequencing errors and genetic variations, making it robust for field use.

### Technical Advantages
Key innovations include:
1. **Reverse-complement invariance** eliminating strand bias
2. **Chromosomal position tracking** for context-aware analysis 
3. **Memory-efficient sequence windowing** (1kb segments) enabling whole-genome analysis on consumer GPUs

### Practical Applications
The model outputs **interpretable attention maps** showing which genetic regions influenced predictions, crucial for tracking drug-resistant strain evolution.