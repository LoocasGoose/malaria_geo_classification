"""
Malaria Geographic Origin Classifier CNN, Advanced. 

This CNN model specializes in identifying the geographic origin of malaria 
samples by analyzing patterns in their DNA sequences. Key features:
- Processes both DNA strands simultaneously
- Focuses on biologically relevant patterns using attention
- Handles real-world sequencing artifacts and variants
- Provides interpretable predictions

Example Usage:
    model = DNACNN(n_classes=25)
    trainer = train(model, train_loader, val_loader, ...)
    evaluate(model, test_loader, ...)

Please read the cnn_README.md file for more information on differences between
this model and cnn_standard.py. 
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from src.data.genomic_sequences import GenomicSequenceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_reverse_complement(x):
    """Create reverse complement of DNA sequences in one-hot encoding.
    
    Args:
        x: One-hot encoded DNA tensor [batch, channels, seq_length]
            Assumes channel order: A=0, C=1, G=2, T=3, N=4
    
    Returns:
        Reverse complemented tensor of same shape
    """
    # Reverse the sequence
    x_rev = torch.flip(x, dims=[2])
    
    # Use tensor indexing for faster complementation
    # A↔T (0↔3), C↔G (1↔2), N (4) stays the same
    x_comp = x_rev[:, [3, 2, 1, 0, 4], :]
    
    return x_comp


class StrandSymmetricConv(nn.Module):
    """Convolutional layer that processes both forward and reverse strands.
    
    This layer performs the same convolution on both the original sequence
    and its reverse complement, then combines the results. This ensures
    the model identifies patterns regardless of which DNA strand they appear on.
    
    Attributes:
        conv: Convolutional layer applied to both strands
        strand_weights: Learnable weights for combining strand-specific features
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        # Learnable weights for combining forward and reverse strands
        # Initialize with equal weights (like averaging)
        self.strand_weights = nn.Parameter(torch.ones(2) / 2.0)
        
    def forward(self, x):
        # Process forward strand
        x_fwd = self.conv(x)
        
        # Process reverse complement
        x_rev = create_reverse_complement(x)
        x_rev = self.conv(x_rev)
        x_rev = torch.flip(x_rev, [2])  # Reverse back to match positions
        
        # Use learned weights to combine strands instead of simple averaging
        # Normalize weights with softmax to ensure they sum to 1
        norm_weights = F.softmax(self.strand_weights, dim=0)
        x = norm_weights[0] * x_fwd + norm_weights[1] * x_rev
        
        return x

class PositionalEncoding(nn.Module):
    """Adds genomic position context to DNA sequences
    
    Helps the model understand where patterns occur in the chromosome,
    important because some genetic features are location-dependent.
    
    Args:
        d_model: Number of positional encoding channels
        max_len: Maximum sequence length to handle
        n_chromosomes: Number of different chromosomes to encode
    
    Input: 
        x: [batch, seq_len, channels] DNA sequence
        positions: [batch] Starting genomic positions
        chromosomes: [batch] Chromosome identifiers
    Output: [batch, seq_len, channels + d_model] Enhanced sequence
    """
    
    def __init__(self, d_model, max_len=10000, n_chromosomes=14):
        super().__init__()
        # Standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Chromosome-specific encoding
        # Allows model to learn chromosome-specific patterns
        self.chrom_embedding = nn.Embedding(n_chromosomes + 1, d_model // 4)  # +1 for unknown chromosome
        
    def forward(self, x, positions=None, chromosomes=None):
        """
        Args:
            x: Tensor [batch, seq_len, channels]
            positions: Absolute genomic positions [batch] or None
            chromosomes: Chromosome identifiers [batch] or None
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Default position if not provided
        if positions is None:
            positions = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            
        # Default chromosome if not provided
        if chromosomes is None:
            chromosomes = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Ensure positions are within bounds
        positions = torch.clamp(positions, 0, self.pe.size(0) - seq_len)
        
        # Position-based encoding - choose the appropriate window for each sequence
        pos_encodings = []
        for i in range(batch_size):
            pos_enc = self.pe[positions[i]:positions[i]+seq_len]
            pos_encodings.append(pos_enc)
        pos_encodings = torch.stack(pos_encodings, dim=0)
        
        # Chromosome-specific encoding
        chrom_enc = self.chrom_embedding(chromosomes)  # [batch, d_model//4]
        
        # Expand chromosome encoding to match sequence length
        chrom_enc = chrom_enc.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine position and chromosome encodings
        combined_encoding = torch.cat([pos_encodings, chrom_enc], dim=2)
        
        # Concatenate with original sequence data
        return torch.cat([x, combined_encoding], dim=2)

class AttentionPooling(nn.Module):
    """Focuses on key regions of DNA sequences
    
    Learns which parts of the DNA sequence are most important for
    determining geographic origin, improving interpretability.
    
    Args:
        in_channels: Input channels from previous layer
    
    Input: [batch, channels, seq_len] DNA features
    Output: [batch, channels] Weighted sequence summary
    """
    
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
            
    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        attn_weights = self.attention(x)  # [batch, 1, seq_len]
        return torch.sum(x * attn_weights, dim=2)  # [batch, channels]

class DenseResidualBlock(nn.Module):
    """Enhanced residual block with dense connections and cross-layer features
    
    Combines ideas from ResNet, DenseNet, and ResNeXt to improve gradient flow
    and feature reuse across multiple scales and layers. Includes gating
    mechanisms for dynamic feature selection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        cardinality: Number of groups for grouped convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, cardinality=4):
        super().__init__()
        # Ensure odd kernel sizes for better symmetric padding
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Split processing into multiple paths (cardinality)
        self.paths = nn.ModuleList()
        path_channels = out_channels // cardinality
        
        for i in range(cardinality):
            # Ensure second kernel is also odd-sized
            second_kernel = (kernel_size // 2) * 2 + 1  # Always odd
            path = nn.Sequential(
                StrandSymmetricConv(in_channels, path_channels, kernel_size, padding='same'),
                nn.BatchNorm1d(path_channels),
                nn.ReLU(),
                StrandSymmetricConv(path_channels, path_channels, second_kernel, padding='same'),
                nn.BatchNorm1d(path_channels),
            )
            self.paths.append(path)
        
        # Cross-path attention for information exchange
        self.cross_path_attn = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fusion layer to combine all features
        total_channels = in_channels + out_channels
        self.fusion = nn.Conv1d(total_channels, out_channels, 1)
        self.bn_fusion = nn.BatchNorm1d(out_channels)
        
        # Edge connections (skip connections for remote layers)
        # These will be used if this block is part of a sequence of blocks
        self.edge_gate = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final activation
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_input=None):
        # First, process each path separately
        path_outputs = []
        for path in self.paths:
            path_outputs.append(path(x))
        
        # Combine path outputs
        path_combined = torch.cat(path_outputs, dim=1)
        
        # Apply cross-path attention
        attn = self.cross_path_attn(path_combined)
        path_combined = path_combined * attn
        
        # Include edge connection if provided
        if edge_input is not None:
            # Gate the edge input to control information flow
            edge_importance = self.edge_gate(edge_input)
            path_combined = path_combined + (edge_input * edge_importance)
        
        # Concatenate with original input (dense connection)
        dense_out = torch.cat([x, path_combined], dim=1)
        
        # Fusion layer to get back to desired channel count
        out = self.fusion(dense_out)
        out = self.bn_fusion(out)
        
        return self.relu(out)

class HierarchicalAttention(nn.Module):
    """Applies attention at multiple scales and combines results
    
    Helps model focus on important features at different resolutions -
    from small motifs to large genomic regions
    
    Args:
        channel_sizes: List of channel dimensions from each conv layer
    """
    def __init__(self, channel_sizes):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            AttentionPooling(channels) for channels in channel_sizes
        ])
        self.fusion = nn.Linear(sum(channel_sizes), channel_sizes[-1])
    
    def forward(self, features):
        # Apply attention at each resolution level
        attention_outputs = []
        for i, feature in enumerate(features):
            attention_outputs.append(self.attention_layers[i](feature))
        
        # Concatenate and fuse multi-scale attention
        combined = torch.cat(attention_outputs, dim=1)
        return self.fusion(combined)

class DNACNN(nn.Module):
    """CNN model for DNA sequence classification.
    
    Features:
        - Hierarchical processing of DNA sequences
        - Strand-symmetric convolutions (optional)
        - Positional encoding for genomic context
        - Attention mechanisms for interpretability
        - Residual connections for gradient flow
    
    Args:
        n_classes: Number of output classes
        seq_length: Input DNA sequence length (default 1000bp)
        n_channels: Input channels (5 for A,C,G,T,N)
        conv_channels: Convolutional filter counts [64, 128, 256]
        kernel_sizes: DNA pattern sizes to detect [15,9,5] 
        fc_sizes: Hidden layer sizes [1024, 512]
        dropout: Dropout rate for regularization (default 0.4)
        conv_type: Type of convolution to use ('symmetric' or 'standard')
    
    Input: [batch, seq_length, 5] DNA sequences
    Output: [batch, n_classes] Country probabilities
    """
    
    def __init__(self, n_classes, seq_length=1000, n_channels=5, 
                 conv_channels=[64, 128, 256], kernel_sizes=[15, 9, 5], 
                 fc_sizes=[1024, 512], dropout=0.4, conv_type='symmetric'):
        """
        Initialize CNN model.
        
        Args:
            n_classes: Number of output classes
            seq_length: Length of input sequences
            n_channels: Number of input channels (5 for one-hot DNA)
            conv_channels: List of channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            fc_sizes: List of hidden units for fully connected layers
            dropout: Dropout probability
            conv_type: Type of convolution ('symmetric' or 'standard')
        """
        super(DNACNN, self).__init__()
        
        # Select convolution type
        self.conv_type = conv_type
        ConvLayer = StrandSymmetricConv if conv_type == 'symmetric' else StandardConv
        
        # Input shape: [batch_size, 1, seq_length, n_channels]
        
        # Add positional encoding
        # Use a smaller dimension for positional information to avoid overwhelming the sequence data
        self.pos_encoding = PositionalEncoding(d_model=16, max_len=seq_length, n_chromosomes=14)
        
        # Convolutional layers with cross-layer connections
        self.conv_layers = nn.ModuleList()
        
        # First conv layer takes input from n_channels + positional channels
        in_channels = n_channels + 16 + 4  # add 4 for chromosome embedding
        
        # Store intermediate outputs for cross-layer connections
        self.cross_connections = []
        
        for i, out_channels in enumerate(conv_channels):
            conv_layer = nn.Sequential(
                DenseResidualBlock(in_channels, out_channels, kernel_sizes[i], cardinality=4),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.conv_layers.append(conv_layer)
            in_channels = out_channels
            seq_length = seq_length // 2  # Update sequence length after pooling
        
        # Calculate flattened size after all convolutions
        self.flattened_size = seq_length * conv_channels[-1]
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = self.flattened_size
        
        for out_features in fc_sizes:
            fc_layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc_layers.append(fc_layer)
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Attention pooling
        self.hierarchical_attn = HierarchicalAttention(conv_channels)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, positions=None, chromosomes=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, n_channels]
            positions: Optional starting positions for sequences [batch_size]
                      If None, assumes positions start at 0
            chromosomes: Optional chromosome identifiers [batch_size]
                        If None, assumes chromosome 0
        
        Returns:
            Tensor of shape [batch_size, n_classes]
        """
        batch_size, seq_len, n_channels = x.shape
        
        # Apply positional encoding with chromosome information
        if positions is None:
            # If no specific positions provided, use simple sequencing from 0
            positions = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Apply positional encoding - first need to ensure proper shape
        # [batch, seq_len, n_channels] -> add positional encoding -> [batch, seq_len, n_channels+d_model]
        x = self.pos_encoding(x, positions, chromosomes)
        
        # Rearrange input dimensions for 1D convolution: [batch, channels, length]
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers
        features = []  # Store outputs from each conv layer
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            features.append(x)
        
        # Apply hierarchical attention
        x = self.hierarchical_attn(features)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Output layer (logits)
        x = self.output_layer(x)
        
        return x


def augment_sequence(seq_tensor, p=0.05):
    """Vectorized sequence augmentation"""
    if torch.rand(1).item() < 0.5:
        mask = torch.rand_like(seq_tensor[:,:,0]) < p
        aug_tensor = seq_tensor.clone()
        # Zero out all channels where mask is True
        aug_tensor[mask, :] = 0
        # Set N channel to 1
        aug_tensor[mask, 4] = 1
        return aug_tensor
    return seq_tensor


def augment_sequence_with_rc(seq_tensor, p=0.5):
    """Apply random reverse-complement transformation.
    
    In genomics, the same motif can appear on either DNA strand, so
    training on both orientations improves model generalization.
    """
    if torch.rand(1).item() < p:
        # Reverse sequence
        rc_tensor = torch.flip(seq_tensor, dims=[1])
        
        # Swap A↔T and G↔C (assuming channel order is A,C,G,T,N)
        AT_swap = rc_tensor[:, :, [3, 2, 1, 0, 4]]
        return AT_swap
    return seq_tensor


def add_positional_noise(seq_tensor, p=0.02):
    """Apply positional noise by shifting sequences slightly."""
    if torch.rand(1).item() < 0.5:
        # Only apply to some sequences in batch
        batch_size = seq_tensor.shape[0]
        # Decide which sequences to modify
        mask = torch.rand(batch_size) < p
        if not mask.any():
            return seq_tensor
            
        # Make a copy to modify
        new_tensor = seq_tensor.clone()
        
        # For each selected sequence, shift by -1, 0, or 1
        for i in range(batch_size):
            if mask[i]:
                shift = torch.randint(-1, 2, (1,)).item()
                if shift != 0:
                    # Shift the sequence left or right
                    new_tensor[i] = torch.roll(new_tensor[i], shifts=shift, dims=0)
                    
                    # Fill in the wrapped values with N (position 4)
                    if shift > 0:
                        new_tensor[i, :shift, :] = 0
                        new_tensor[i, :shift, 4] = 1  # N base
                    else:
                        new_tensor[i, shift:, :] = 0
                        new_tensor[i, shift:, 4] = 1  # N base
                        
        return new_tensor
    return seq_tensor


def simulate_variants(seq_tensor, p_snp=0.01, p_ins=0.005, p_del=0.005):
    """Simulate genomic variants (SNPs, insertions, deletions) in sequences.
    
    Args:
        seq_tensor: One-hot encoded DNA tensor [batch, seq_len, channels]
        p_snp: Probability of introducing a SNP at any position
        p_ins: Probability of introducing an insertion at any position
        p_del: Probability of introducing a deletion at any position
        
    Returns:
        Tensor with simulated variants
    """
    if torch.rand(1).item() > 0.3:  # Only apply to ~30% of batches
        return seq_tensor
        
    batch_size, seq_len, n_channels = seq_tensor.shape
    result = seq_tensor.clone()
    
    # Process each sequence in the batch
    for b in range(batch_size):
        # 1. Simulate SNPs (substitutions)
        if p_snp > 0:
            # Create a mask for positions to modify
            snp_mask = torch.rand(seq_len) < p_snp
            snp_positions = torch.nonzero(snp_mask).squeeze(-1)
            
            for pos in snp_positions:
                # Get current base (argmax of one-hot encoding)
                current_base = torch.argmax(seq_tensor[b, pos]).item()
                if current_base == 4:  # Don't modify N bases
                    continue
                    
                # Choose a different base (excluding N)
                new_base = torch.randint(0, 4, (1,)).item()
                while new_base == current_base:
                    new_base = torch.randint(0, 4, (1,)).item()
                
                # Update the one-hot encoding
                result[b, pos] = torch.zeros(n_channels)
                result[b, pos, new_base] = 1.0
                
        # 2. Simulate deletions
        if p_del > 0:
            # Choose positions for deletions
            del_mask = torch.rand(seq_len) < p_del
            del_positions = torch.nonzero(del_mask).squeeze(-1)
            
            for pos in del_positions:
                # Skip if too close to the end
                if pos >= seq_len - 1:
                    continue
                
                # Shift sequence to simulate deletion
                result[b, pos:-1] = result[b, pos+1:].clone()
                # Fill the end with N
                result[b, -1] = torch.zeros(n_channels)
                result[b, -1, 4] = 1.0  # N base
                
        # 3. Simulate insertions
        if p_ins > 0:
            # Choose positions for insertions
            ins_mask = torch.rand(seq_len) < p_ins
            ins_positions = torch.nonzero(ins_mask).squeeze(-1)
            
            for pos in reversed(ins_positions):  # Process in reverse to avoid position shifting
                # Skip if too close to the end
                if pos >= seq_len - 1:
                    continue
                    
                # Shift sequence to make room for insertion
                result[b, pos+1:] = result[b, pos:-1].clone()
                
                # Choose a random base to insert (A, C, G, or T)
                new_base = torch.randint(0, 4, (1,)).item()
                result[b, pos] = torch.zeros(n_channels)
                result[b, pos, new_base] = 1.0
    
    return result


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0, verbose=True, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.best_score:.6f} → {-val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, n_epochs=30, 
          device='cuda', model_dir='models', early_stopping_patience=5, mixed_precision=True):
    """Train a genomic CNN model with advanced techniques.
    
    This training function incorporates:
    - Early stopping to prevent overfitting
    - Learning rate scheduling
    - Mixed precision for faster training
    - Checkpointing to save best models
    - Comprehensive metrics tracking
    - Data augmentation including variant simulation
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        n_epochs: Maximum number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        model_dir: Directory to save model checkpoints
        early_stopping_patience: Epochs to wait before stopping
        mixed_precision: Use GPU acceleration (recommended)
        
    Returns:
        dict: Training history with metrics
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, path=os.path.join(model_dir, 'best_model.pt'))
    
    # Initialize scaler for mixed precision training
    scaler = GradScaler() if mixed_precision and device == 'cuda' else None
    
    # Get current timestamp for model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add learning rate warmup
    warmup_epochs = 3
    warmup_factor = 1.0 / warmup_epochs
    
    # Training loop
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimizer.defaults['lr'] * (epoch + 1) * warmup_factor
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Get data
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Apply data augmentation with variant simulation
                if model.training:  # Check if the model is in training mode
                    # Apply sequence augmentations
                    sequences = augment_sequence(sequences)
                    sequences = augment_sequence_with_rc(sequences)
                    sequences = add_positional_noise(sequences)
                    sequences = simulate_variants(sequences, p_snp=0.01, p_ins=0.005, p_del=0.005)
                
                # Mixed precision forward pass
                if mixed_precision and device == 'cuda':
                    with autocast():
                        outputs = model(sequences)
                        loss = criterion(outputs, labels)
                    
                    # Scale loss and backprop
                    scaler.scale(loss).backward()
                    # Add gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Original code path
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                
                # Track loss and predictions
                train_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                continue  # Skip this batch
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                # Track loss and predictions
                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update learning rate scheduler if provided
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        # Print progress
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{n_epochs} - {epoch_time:.2f}s - "
                     f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                     f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    # Load best model (saved during early stopping)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'cnn_advanced.pt')))
    
    # Also save final model (which may be different from best)
    torch.save(model.state_dict(), os.path.join(model_dir, f'final_model_{timestamp}.pt'))
    
    # Save training history
    with open(os.path.join(model_dir, f'history_{timestamp}.json'), 'w') as f:
        json.dump(history, f)
    
    return model, history


def evaluate(model, test_loader, criterion, device):
    """Evaluate model performance on test data"""
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate additional metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_precision = precision_score(test_labels, test_preds, average='weighted')
    test_recall = recall_score(test_labels, test_preds, average='weighted')
    
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall)
    }
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1: {test_f1:.4f}")
    logging.info(f"Test Precision: {test_precision:.4f}")
    logging.info(f"Test Recall: {test_recall:.4f}")
    
    return metrics


def get_batch_size(device):
    """Determine appropriate batch size based on available memory"""
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
        if gpu_mem < 8:
            return 16
        elif gpu_mem < 16:
            return 32
        else:
            return 64
    return 32  # Default for CPU


def main():
    """Run full model training and evaluation"""
    # Import visualization function from evaluator module
    from torch.utils.data import DataLoader
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load datasets
    logging.info("Loading datasets...")
    
    # Configure dataset parameters
    window_size = 1000  # 1kb windows
    stride = 500  # 50% overlap
    
    # Create datasets
    train_dataset = GenomicSequenceDataset(
        split_dir="data/split",
        split_type="train",
        window_size=window_size,
        stride=stride,
        cache_size=128
    )
    
    val_dataset = GenomicSequenceDataset(
        split_dir="data/split",
        split_type="val",
        window_size=window_size,
        stride=stride,
        cache_size=128
    )
    
    test_dataset = GenomicSequenceDataset(
        split_dir="data/split",
        split_type="test",
        window_size=window_size,
        stride=stride,
        cache_size=128
    )
    
    # Sample for testing if desired (comment out for full training)
    # train_dataset.sample_regions(n_regions=1000)
    # val_dataset.sample_regions(n_regions=100)
    # test_dataset.sample_regions(n_regions=100)
    
    # Create dataloaders
    batch_size = get_batch_size(device)
    logging.info(f"Using batch size: {batch_size} based on available memory")
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_dataset.get_dataloader(batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Calculate number of classes
    n_classes = len(train_dataset.encoder.classes_)
    logging.info(f"Number of classes: {n_classes}")
    
    # Calculate class weights for balanced loss
    if hasattr(train_dataset, 'labels'):
        class_counts = np.bincount(train_dataset.labels)
        class_weights = torch.FloatTensor(1.0 / class_counts)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = class_weights.to(device)
        logging.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None

    # Label smoothing
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, x, target):
            log_probs = F.log_softmax(x, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
            return loss.mean()

    # Use in training:
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Create model
    model_symmetric = DNACNN(
        n_classes=n_classes,
        seq_length=window_size,
        n_channels=5,  # A, C, G, T, N
        conv_channels=[64, 128, 256],
        kernel_sizes=[15, 9, 5],
        fc_sizes=[1024, 512],
        dropout=0.4,
        conv_type='symmetric'
    )
    
    # Add multi-GPU support
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model_symmetric = nn.DataParallel(model_symmetric)
    
    # Define optimizer
    optimizer = optim.AdamW(model_symmetric.parameters(), lr=0.001, weight_decay=1e-4)  # Increased weight decay
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training
    logging.info("Starting training...")
    model_symmetric, history = train(
        model=model_symmetric,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=30,
        device=device,
        model_dir='models',
        early_stopping_patience=5,
        mixed_precision=True
    )
    
    # Evaluation
    logging.info("Evaluating model on test set...")
    metrics_symmetric = evaluate(
        model=model_symmetric,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    logging.info("Training and evaluation complete!")

    # Without equivariance
    model_standard = DNACNN(
        n_classes=n_classes,
        seq_length=window_size,
        n_channels=5,  # A, C, G, T, N
        conv_channels=[64, 128, 256],
        kernel_sizes=[15, 9, 5],
        fc_sizes=[1024, 512],
        dropout=0.4,
        conv_type='standard'
    )
    
    # Add multi-GPU support
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model_standard = nn.DataParallel(model_standard)
    
    # Define optimizer
    optimizer = optim.AdamW(model_standard.parameters(), lr=0.001, weight_decay=1e-4)  # Increased weight decay
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training
    logging.info("Starting training without equivariance...")
    model_standard, history = train(
        model=model_standard,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=30,
        device=device,
        model_dir='models',
        early_stopping_patience=5,
        mixed_precision=True
    )
    
    # Evaluation
    logging.info("Evaluating model on test set without equivariance...")
    metrics = evaluate(
        model=model_standard,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    logging.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()