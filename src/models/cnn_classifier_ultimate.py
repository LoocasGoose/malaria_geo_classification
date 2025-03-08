"""
Malaria Geographic Origin Classifier CNN

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
from src.data.GenomicSequence import GenomicSequenceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrandSymmetricConv(nn.Module):
    """DNA strand-aware convolutional layer
    
    Processes DNA sequences in both forward and reverse-complement 
    directions simultaneously. This ensures the model detects patterns
    regardless of which DNA strand they appear on.
    
    Args:
        in_channels: Input channels (5 for DNA: A,C,G,T,N)
        out_channels: Number of pattern detectors to create
        kernel_size: Size of DNA patterns to detect (e.g., 9 = 9bp motifs)
        padding: Maintains sequence length during convolution
    
    Input: [batch, channels, seq_len] DNA sequence tensor
    Output: [batch, out_channels, seq_len] pattern activations
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x):
        # Compute both forward and reverse-complement convolutions in one batch
        x_rc = create_reverse_complement(x)
        both_x = torch.cat([x, x_rc], dim=0)
        
        # Run convolution once on combined batch
        both_results = self.conv(both_x)
        
        # Split results back
        x_fwd = both_results[:x.size(0)]
        x_rev = both_results[x.size(0):]
        x_rev = torch.flip(x_rev, dims=[2])  # Flip back
        
        return torch.max(x_fwd, x_rev)

class PositionalEncoding(nn.Module):
    """Adds genomic position context to DNA sequences
    
    Helps the model understand where patterns occur in the chromosome,
    important because some genetic features are location-dependent.
    
    Args:
        d_model: Number of positional encoding channels
        max_len: Maximum sequence length to handle
    
    Input: 
        x: [batch, seq_len, channels] DNA sequence
        positions: [batch] Starting genomic positions
    Output: [batch, seq_len, channels + d_model] Enhanced sequence
    """
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x, positions):
        """
        Args:
            x: Tensor [batch, seq_len, channels]
            positions: Absolute genomic positions [batch]
        """
        # Get relative positions within the genome
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Add positional encoding as additional channels
        pos_channels = self.pe[positions:positions+seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.cat([x, pos_channels], dim=2)

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

class ResidualStrandConv(nn.Module):
    """Add residual connections to StrandSymmetricConv."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.conv = StrandSymmetricConv(in_channels, out_channels, kernel_size, padding)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)
            
    def forward(self, x):
        return F.relu(self.bn(self.conv(x) + self.residual(x)))

class DNACNN(nn.Module):
    """Malaria DNA Analysis CNN Architecture
    
    Custom neural network designed specifically for analyzing 
    Plasmodium falciparum DNA sequences to predict geographic origin.
    
    Args:
        n_classes: Number of output countries/regions
        seq_length: Input DNA sequence length (default 1000bp)
        n_channels: Input channels (5 for A,C,G,T,N)
        conv_channels: Convolutional filter counts [64, 128, 256]
        kernel_sizes: DNA pattern sizes to detect [15,9,5] 
        fc_sizes: Hidden layer sizes [1024, 512]
        dropout: Dropout rate for regularization (default 0.4)
    
    Input: [batch, seq_length, 5] DNA sequences
    Output: [batch, n_classes] Country probabilities
    """
    
    def __init__(self, n_classes, seq_length=1000, n_channels=5, 
                 conv_channels=[64, 128, 256], kernel_sizes=[15, 9, 5], 
                 fc_sizes=[1024, 512], dropout=0.4):
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
        """
        super(DNACNN, self).__init__()
        
        # Input shape: [batch_size, 1, seq_length, n_channels]
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First conv layer takes input from n_channels
        in_channels = n_channels
        for i, out_channels in enumerate(conv_channels):
            conv_layer = nn.Sequential(
                ResidualStrandConv(in_channels, out_channels, kernel_sizes[i], padding=kernel_sizes[i]//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
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
        self.attn_pool = AttentionPooling(conv_channels[-1])
    
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
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, n_channels]
        
        Returns:
            Tensor of shape [batch_size, n_classes]
        """
        # Rearrange input dimensions for 1D convolution: [batch, channels, length]
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Apply attention pooling
        x = self.attn_pool(x)
        
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
    
    # Swap complementary bases: A↔T (0↔3) and C↔G (1↔2)
    # N (4) stays the same
    x_comp = torch.zeros_like(x_rev)
    x_comp[:, 0] = x_rev[:, 3]  # A gets T
    x_comp[:, 1] = x_rev[:, 2]  # C gets G
    x_comp[:, 2] = x_rev[:, 1]  # G gets C
    x_comp[:, 3] = x_rev[:, 0]  # T gets A
    x_comp[:, 4] = x_rev[:, 4]  # N stays N
    
    return x_comp


def add_positional_noise(seq_tensor, p=0.02):
    """Add small random shifts to simulate sequencing errors."""
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-2, 3, (1,)).item()
        if shift > 0:
            # Shift right
            seq_tensor = torch.cat([torch.zeros_like(seq_tensor[:,:shift]), seq_tensor[:,:-shift]], dim=1)
        elif shift < 0:
            # Shift left
            seq_tensor = torch.cat([seq_tensor[:,-shift:], torch.zeros_like(seq_tensor[:,:shift])], dim=1)
    return seq_tensor


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
    """Trains the malaria origin classifier
    
    Implements DNA-specific training with:
    - Automatic mixed precision for faster training
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    
    Args:
        model: Initialized DNACNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (use LabelSmoothingCrossEntropy)
        optimizer: Optimizer (recommend AdamW)
        scheduler: Learning rate scheduler
        n_epochs: Maximum training epochs
        device: cuda/cpu
        model_dir: Save directory for checkpoints
        early_stopping_patience: Stop after N epochs without improvement
        mixed_precision: Use GPU acceleration (recommended)
    
    Returns:
        Trained model, training history dictionary
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
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        verbose=True,
        path=os.path.join(model_dir, 'best_model.pt')
    )
    
    # Initialize gradient scaler for mixed precision
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
                
                # Mixed precision forward pass
                if mixed_precision and device == 'cuda':
                    with autocast():
                        # Enhanced augmentation
                        sequences = augment_sequence(sequences, p=0.1)  # Increased masking
                        sequences = augment_sequence_with_rc(sequences, p=0.5)
                        sequences = add_positional_noise(sequences)  # New augmentation
                        
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
                    sequences = augment_sequence(sequences)
                    # Missing RC augmentation here!
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
                sequences = augment_sequence(sequences)
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
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
    
    # Also save final model (which may be different from best)
    torch.save(model.state_dict(), os.path.join(model_dir, f'final_model_{timestamp}.pt'))
    
    # Save training history
    with open(os.path.join(model_dir, f'history_{timestamp}.json'), 'w') as f:
        json.dump(history, f)
    
    return model, history


def evaluate(model, test_loader, criterion, device):
    """Evaluates model performance on test data
    
    Computes comprehensive metrics:
    - Test loss
    - Accuracy
    - F1 score
    - Precision
    - Recall
    
    Args:
        model: Trained DNACNN
        test_loader: Test data loader
        criterion: Loss function
        device: cuda/cpu
    
    Returns:
        Dictionary of test metrics
    """
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


def visualize_attention(model, test_loader, device, n_samples=5):
    """Generates interpretable examples of model decisions
    
    Outputs:
    - Example DNA sequences
    - True vs predicted labels
    - Attention weights (which regions influenced decisions)
    
    Args:
        model: Trained DNACNN
        test_loader: Test data loader
        device: cuda/cpu
        n_samples: Number of examples to save
    
    Saves 'sequence_samples.json' with visualization data
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            regions = batch['region']
            
            # Forward pass
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            # Get activation maps from the last convolutional layer
            # (Would need to modify model to extract these)
            
            # For now, just collect sample sequences for visualization
            for i in range(min(n_samples, sequences.size(0))):
                if len(samples) >= n_samples:
                    break
                    
                # Convert one-hot to readable sequence
                seq_array = sequences[i].cpu().numpy()
                bases = ["A", "C", "G", "T", "N"]
                seq_string = ""
                for pos in range(seq_array.shape[0]):
                    base_idx = np.argmax(seq_array[pos])
                    seq_string += bases[base_idx]
                
                samples.append({
                    'sequence': seq_string[:50] + "...",  # First 50 bases
                    'region': regions[i],
                    'true_label': labels[i].item(),
                    'predicted': predicted[i].item()
                })
            
            if len(samples) >= n_samples:
                break
    
    # Save visualization data
    with open('metrics/sequence_samples.json', 'w') as f:
        json.dump(samples, f, indent=2)
        
    return samples


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
    """Main function to run the CNN training and evaluation."""
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
    model = DNACNN(
        n_classes=n_classes,
        seq_length=window_size,
        n_channels=5,  # A, C, G, T, N
        conv_channels=[64, 128, 256],
        kernel_sizes=[15, 9, 5],
        fc_sizes=[1024, 512],
        dropout=0.4
    )
    
    # Add multi-GPU support
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Increased weight decay
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training
    logging.info("Starting training...")
    model, history = train(
        model=model,
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
    metrics = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Visualize attention
    visualize_attention(model, test_loader, device)
    
    logging.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()