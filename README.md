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

1. **Data acquisition**: Access the MalariaGEN Pf7 dataset via the `malariagen_data` API. Need to request data access to their Google Cloud access. 
2. **Quality filtering**: Select high-quality samples based on metadata thresholds
3. **Feature extraction**: Process genetic variants into TF-IDF weighted feature vectors
4. **Data splitting**: Partition data into training, validation, and test sets with stratification
5. **Sequence windowing**: Generate fixed-length windows (1kb) from reference genomes with variants applied

## Configuration

The project uses a centralized configuration system for easy parameter management:

- **config.yml**: Central configuration file containing all parameters for data processing, model training, and evaluation. This allows quick experimentation without modifying code.
- **environment.yml**: Conda environment specification for reproducible environment setup.
- **utils.py**: Common utility functions including configuration loading, logging setup, and performance measurement.

To customize the project for your needs, simply modify the parameters in `config.yml` rather than changing code directly.

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

## Model Evaluation and Comparison

The project includes a comprehensive evaluation framework to compare model performance across multiple dimensions:

### Performance Metrics

Models are evaluated using a range of metrics:
- **Accuracy**: Overall classification correctness
- **Precision/Recall/F1**: Class-weighted metrics to handle potential geographic imbalance
- **Inference Time**: Computational efficiency for practical deployment
- **ROC and Precision-Recall Curves**: For multi-class evaluation

### Strand Symmetry Analysis

A dedicated module (`compare_strand_symmetry.py`) analyzes the impact of reverse complement equivariance:
- Compares advanced CNN variants with and without strand symmetry
- Quantifies performance differences across metrics
- Identifies geographic regions most affected by strand-specific processing
- Provides visualizations of relative improvements

### Complexity-Performance Tradeoff

The evaluation framework includes tools to analyze the relationship between model complexity and performance:
- Model size and parameter count comparisons
- Inference speed vs. accuracy tradeoffs
- Efficiency scores (accuracy per ms of inference time)
- Per-class performance analysis to identify where complexity helps most

## Visualization Tools

### Genomic Region Visualization

Interactive HTML visualizations show which genomic regions influence classification decisions:
- **Attention Heatmaps**: Highlighting important sequence positions
- **Saliency Maps**: Gradient-based visualization of input importance
- **Feature Maps**: Activation patterns across convolutional layers
- **Motif Discovery**: Potential binding sites or functional elements derived from high-attention regions

### Performance Visualization

Comprehensive visualization tools include:
- **Confusion Matrices**: Both count and percentage based
- **ROC Curves**: For all models with AUC comparison
- **Per-Class Performance**: Barplots of metrics by geographic region
- **Training History**: Loss and accuracy curves over epochs
- **Model Architecture Comparison**: Size and parameter visualizations

## Performance Analysis

(THE PF7 DATASET ACCESS HAS NOT GET BEEN GRANTED TO ME BY MALARIAGEN AND THUS I CANNOT RUN THE MODELS)

### Geographic Classification Patterns

### Model-Specific Strengths

### Deployment Recommendations

## Project Structure

```
malaria-classification/
├── data/                      # Data storage
│   ├── processed/             # Preprocessed data
│   ├── reference/             # Reference genomic sequences
│   └── split/                 # Train/val/test splits
├── models/                    # Saved model weights
├── src/
│   ├── data/                  # Data processing modules
│   │   ├── preprocess.py      # Data acquisition and preprocessing
│   │   ├── data_splitter.py   # Train/val/test splitting
│   │   └── genomic_sequences.py # PyTorch dataset for sequences
│   ├── models/                # Model definitions
│   │   ├── naive_bayes.py     # Multinomial Naive Bayes
│   │   ├── cnn_standard.py    # Standard CNN architecture
│   │   ├── cnn_advanced.py    # Advanced CNN architecture
│   │   └── train.py           # Training pipeline
│   ├── evaluation/            # Evaluation tools
│   │   ├── model_evaluator.py # Model evaluation and visualization
│   │   ├── model_comparison.py # Cross-model comparison
│   │   └── compare_strand_symmetry.py # Strand symmetry analysis
│   └── visualization/         # Visualization utilities
│       └── performance_visualizer.py # Performance visualization tools
├── reports/                   # Analysis reports and figures
├── README.md                  # Project overview
├── TECHNICAL_DETAILS.md       # Detailed technical documentation
├── QUICKSTART.md              # Getting started guide
├── config.yml                 # Centralized configuration
├── environment.yml            # Conda environment specification
└── requirements.txt           # Project dependencies
```

## Getting Started

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions on:
- Setting up the environment
- Downloading and preprocessing data
- Training models
- Evaluating performance
- Generating visualizations

## Citation and References

- MalariaGEN Pf7 dataset: https://www.malariagen.net/parasite/pf7
- [Additional references will be added here]
