# Technical Documentation: Malaria Geographic Classification

This document provides detailed technical information about the data processing pipeline and model architectures used in the Malaria Geographic Classification project.

## Data Processing Pipeline

### 1. Data Acquisition and Preprocessing (`preprocess.py`)

The preprocessing module handles the initial data acquisition and preparation:

- **Data Source**: MalariaGEN Pf7 dataset accessed via the `malariagen_data` API
- **Quality Filtering**: Samples are filtered based on metadata quality thresholds:
  - Only samples with "QC pass" = True
  - Minimum "% callable" threshold (default: 0.5)
  - Minimum samples per country (default: 50)
- **Variant Processing**:
  - Extracts variant data from selected chromosomes (default: chromosomes 1, 4, 7, 13)
  - Converts variants to binary presence/absence matrices
  - Applies TF-IDF transformation to weight features using batched processing for memory efficiency
- **Label Encoding**: Geographic origins (countries) are encoded as numeric labels

Key design choices:
- Selected chromosomes are limited to reduce dimensionality while preserving signal
- TF-IDF transformation highlights region-specific variants while downweighting common variants
- Batched processing enables handling large genomic datasets with limited memory

### 2. Data Splitting (`data_splitter.py`)

The data splitter module handles partitioning the dataset:

- **Stratified Splitting**: Ensures balanced representation of each geographic region in training, validation, and test sets
- **Split Verification**: Checks that each class has sufficient samples in each split
- **Persistence**: Saves splits to disk in optimized formats (sparse matrices for features, numpy arrays for labels)

### 3. Genomic Sequence Dataset (`genomic_sequences.py`)

This custom PyTorch Dataset class handles DNA sequence processing:

- **Window Generation**: Creates fixed-length windows (default: 1kb) from reference genomes
- **Variant Application**: Applies sample-specific variants to reference sequences
- **Memory Optimization**:
  - Implements caching for frequently accessed sequences
  - Uses memory-efficient sequence representation
  - Processes sequences on-demand rather than loading all at once
- **Data Augmentation**: Supports reverse complement generation and other biological augmentations
- **One-Hot Encoding**: Converts DNA sequences (A/C/G/T/N) to one-hot encoded tensors

## Model Architectures

### 1. Multinomial Naive Bayes (`naive_bayes.py`)

A probabilistic classifier well-suited for sparse, high-dimensional genomic data:

- **Pipeline Architecture**:
  - Feature selection using chi-squared test to reduce dimensionality
  - Multinomial Naive Bayes classifier with tuned alpha (smoothing)
- **Hyperparameter Optimization**:
  - Randomized search cross-validation for feature count and smoothing parameters
  - Dynamic k values based on dataset size
  - Class-weighted metrics for handling imbalanced data
- **Performance Evaluation**:
  - Comprehensive metrics including per-class precision, recall, and F1 score
  - Confusion matrix analysis
  - Inference speed benchmarking

### 2. Standard CNN (`cnn_standard.py`)

A CNN architecture designed specifically for DNA sequence classification:

- **Biological Awareness**:
  - Strand-symmetric convolutions process both DNA strands simultaneously
  - Reverse complement handling ensures pattern detection regardless of strand orientation
- **Architecture Components**:
  - Positional encoding (64 dimensions) for genomic context
  - Residual connections with batch normalization for stable training
  - Three convolutional blocks with increasing filter counts (32, 64, 128)
  - Attention pooling to focus on important sequence regions
  - Fully connected layers with dropout for regularization
- **Training Optimizations**:
  - Label smoothing for regularization
  - Gradient clipping to prevent exploding gradients
  - Early stopping with patience
  - Learning rate scheduling

### 3. Advanced CNN (`cnn_advanced.py`)

An enhanced version of the standard CNN with sophisticated improvements:

- **Advanced Skip Connections**:
  - `DenseResidualBlock` replaces `ResidualStrandConv`
  - Incorporates concepts from ResNet, DenseNet, and ResNeXt
  - Uses cardinality for better feature extraction
- **Enhanced Attention**:
  - `HierarchicalAttention` applies attention at multiple layers
  - Improved focus on biologically significant regions
- **Improved Positional Encoding**:
  - Chromosome-aware encoding with larger context window (10,000 bp)
  - Better representation of genomic position
- **Data Augmentation**:
  - Variant simulation (SNPs, insertions, deletions)
  - Multiple sequence augmentation strategies
  - Positional noise to simulate sequencing errors
- **Advanced Training**:
  - Mixed precision training for memory efficiency
  - Optuna hyperparameter optimization
  - Dynamic batch sizing based on available hardware
  - Checkpoint management for training recovery

## Training Pipeline (`train.py`)

The training module orchestrates the model training process:

- **Model Selection**: Supports training Naive Bayes, standard CNN, or advanced CNN
- **Logging and Checkpointing**: Comprehensive logging and model checkpointing
- **Hyperparameter Optimization**: Optional hyperparameter tuning with Optuna
- **Evaluation**: Consistent evaluation metrics across model types
- **Resource Management**: Dynamic resource allocation based on available hardware

## Implementation Details

### Memory Efficiency Techniques

- **Sparse Matrix Representation**: For variant features
- **On-demand Sequence Generation**: Instead of loading all sequences into memory
- **Caching**: LRU cache for frequently accessed reference sequences
- **Batched Processing**: For TF-IDF transformation and other operations
- **Mixed Precision Training**: For CNN models to reduce memory footprint

### Biological Considerations

- **Strand Symmetry**: DNA's double-stranded nature is handled via strand-symmetric convolutions
- **Variant Handling**: Properly models SNPs, insertions, and deletions
- **Sequence Context**: Positional encoding captures genomic context
- **Attention Mechanisms**: Focus on biologically significant regions 