# Quick Start Guide

This guide will help you get started with the Malaria Geographic Classification project, from installation to training your first model.

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for CNN training)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/LoocasGoose/malaria-classification.git
cd malaria-classification

# Create a conda environment
conda create -n malaria python=3.10
conda activate malaria

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. Download and Preprocess Data

The project uses the MalariaGEN Pf7 dataset, which will be automatically downloaded when running the preprocessing script:

```bash
# Create data directories
mkdir -p data/processed data/split data/reference

# Run preprocessing
python src/data/preprocess.py
```

This will:
- Download the Pf7 dataset from MalariaGEN
- Filter samples based on quality thresholds
- Extract variant information
- Apply TF-IDF transformation
- Save processed data to `data/processed/`

### 2. Split Data

Split the preprocessed data into training, validation, and test sets:

```bash
python src/data/data_splitter.py
```

This will create stratified splits and save them to `data/split/`.

## Model Training

### Train Naive Bayes Model

```bash
python src/models/train.py --model naive_bayes
```

### Train Standard CNN Model

```bash
python src/models/train.py --model cnn_standard --epochs 30 --batch_size 128 --learning_rate 0.001
```

### Train Advanced CNN Model

```bash
python src/models/train.py --model cnn_advanced --epochs 50 --batch_size 64 --learning_rate 0.0005 --mixed_precision
```

## Evaluation

Evaluate a trained model on the test set:

```bash
python src/evaluation/evaluate.py --model_path models/cnn_advanced.pt
```

## Prediction

Make predictions on new samples:

```bash
python src/predict.py --model_path models/cnn_advanced.pt --input_file path/to/sequences.fasta
```

## Visualization

Generate visualizations of model performance and attention maps:

```bash
python src/visualization/visualize.py --model_path models/cnn_advanced.pt
```

## Advanced Usage

### Hyperparameter Optimization

```bash
python src/models/train.py --model cnn_advanced --hpo --n_trials 20
```

### Custom Training Configuration

You can customize training parameters:

```bash
python src/models/train.py --model cnn_advanced \
    --conv_channels 64 128 256 \
    --kernel_sizes 15 9 5 \
    --fc_sizes 1024 512 \
    --dropout 0.4 \
    --early_stopping 10
```

## Troubleshooting

### Memory Issues

If you encounter memory issues during training:

1. Reduce batch size: `--batch_size 32`
2. Enable mixed precision: `--mixed_precision`
3. Use a subset of chromosomes: `--chromosomes Pf3D7_01_v3 Pf3D7_04_v3`

### CUDA Out of Memory

For CNN training on GPUs with limited VRAM:

1. Reduce model size: `--conv_channels 32 64 128 --fc_sizes 512 256`
2. Use smaller sequence windows: `--window_size 500 --stride 250`
3. Enable gradient accumulation: `--grad_accum_steps 2` 