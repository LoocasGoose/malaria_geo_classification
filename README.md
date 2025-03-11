# Mapping Malaria: Geographic Classification of Plasmodium falciparum Using Deep Learning

## Project Overview

This project uses machine learning to classify *Plasmodium falciparum* (malaria parasite) DNA sequences by their geographic origin. By analyzing genomic signatures, we can identify region-specific patterns that reflect parasite migration, evolution, and local adaptation - potentially informing public health interventions and epidemiological monitoring.

The approach combines traditional machine learning (Multinomial Naive Bayes) with advanced deep learning (Convolutional Neural Networks) to detect subtle geographic signatures in parasite genomes from the MalariaGEN Pf7 dataset, which includes over 20,000 samples from 33 countries.

## Key Features

- **Biologically-aware sequence processing**: Custom handling of DNA's double-stranded nature with strand-symmetric convolutions
- **Advanced CNN architectures**: Comparing standard CNN with enhanced versions featuring residual connections, attention mechanisms, and positional encoding
- **TF-IDF feature extraction**: Converting genomic variants to weighted feature vectors for improved signal detection
- **Memory-efficient implementation**: Specialized techniques for handling large genomic datasets on consumer GPUs
- **Interpretable predictions**: Attention visualization reveals which genomic regions influence classification decisions

## Data Processing Pipeline

1. **Data acquisition**: Access the MalariaGEN Pf7 dataset via the `malariagen_data` API
2. **Quality filtering**: Select high-quality samples based on metadata thresholds
3. **Feature extraction**: Process genetic variants into TF-IDF weighted feature vectors
4. **Data splitting**: Partition data into training, validation, and test sets with stratification
5. **Sequence windowing**: Generate fixed-length windows (1kb) from reference genomes with variants applied

## Models

### Multinomial Naive Bayes

A probabilistic classifier that models genomic variants using a multinomial distribution. Despite its simplicity, it performs well on high-dimensional, sparse genomic data while providing interpretable outputs and fast training/inference times. Key features:

- Feature selection using chi-squared test to reduce dimensionality
- Hyperparameter tuning via randomized search cross-validation
- Efficient handling of sparse genomic feature matrices
- Comprehensive performance evaluation with class-weighted metrics

### CNN Standard

A CNN architecture specially designed for DNA sequence classification with biological awareness:

- Residual connections for stable gradient flow
- Batch normalization for training stability
- Positional encoding for genomic context
- Dropout layers for regularization
- Strand-symmetric convolutions for biological relevance
- Attention mechanism to focus on important regions

The model processes both DNA strands simultaneously via strand-symmetric convolutions, ensuring it detects patterns regardless of which DNA strand they appear on.

### CNN Advanced

An enhanced version that builds upon the standard CNN with sophisticated architectural improvements:

- **Skip Connections**: Upgraded from `ResidualStrandConv` to `DenseResidualBlock` using concepts from ResNet, DenseNet, and ResNeXt with cardinality
- **Attention Mechanism**: Enhanced single-layer `AttentionPooling` to `HierarchicalAttention` applied at multiple layers
- **Positional Encoding**: Improved from simple encoding to chromosome-aware encoding with larger context window (max_len=10000)
- **Variant Simulation**: Added functionality to model SNPs, insertions, and deletions
- **Sequence Augmentation**: Expanded from simple reverse complement to multiple strategies including general mutation and positional noise
- **Information Fusion**: Upgraded from simple averaging weights to learnable weights for combining strand-specific features

Additional improvements include Optuna hyperparameter tuning, mixed precision training, checkpoint management, dynamic batch sizing, and advanced visualization capabilities.

## Getting Started

[Installation and usage instructions will be added here]

## Citation and References

- MalariaGEN Pf7 dataset: https://www.malariagen.net/parasite/pf7
- [Additional references will be added here]
