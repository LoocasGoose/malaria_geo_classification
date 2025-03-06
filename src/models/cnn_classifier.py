'''
cnn_classifier.py
~~~~~~~~~~

Purpose: 
    Define, train, and evaluate the CNN model for genomic classification.
Components:
    CNN architecture for processing DNA sequences
    Training and evaluation functions
    Metrics tracking and model saving
Output: 
    Saves the trained model and performance metrics
'''

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.GenomicSequence import GenomicSequenceDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrandSymmetricConv(nn.Module):
    """Convolutional layer that's invariant to DNA strand direction."""
    
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
    """Adds positional information about genomic coordinates."""
    
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
    """Attention-guided pooling to focus on informative regions."""
    
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
    """CNN architecture for genomic sequence classification."""
    
    def __init__(self, n_classes, seq_length=1000, n_channels=5, 
                 conv_channels=[32, 64, 128], kernel_sizes=[11, 7, 3], 
                 fc_sizes=[512, 256], dropout=0.3):
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


def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, n_epochs=30, 
          device='cuda', model_dir='models', early_stopping=5, mixed_precision=True):
    """
    Train the CNN model.
    
    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        n_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        model_dir: Directory to save models
        early_stopping: Number of epochs to wait before early stopping
        mixed_precision: Whether to use mixed precision training
    
    Returns:
        Trained model and training history
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement = 0
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if mixed_precision and device == 'cuda' else None
    
    # Training loop
    for epoch in range(n_epochs):
        start_time = time.time()
        
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
                        sequences = augment_sequence(sequences)
                        sequences = augment_sequence_with_rc(sequences)  # RC aug applied
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
        
        # Print progress
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{n_epochs} - {epoch_time:.2f}s - "
                     f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                     f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(model_dir, 'cnn_model_best.pt'))
            
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= early_stopping:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }, os.path.join(model_dir, 'cnn_model_final.pt'))
    
    # Save training history
    with open(os.path.join(model_dir, 'cnn_training_history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'cnn_training_curves.png'))
    
    logging.info(f"Best model saved at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    return model, history


def evaluate(model, test_loader, criterion, device='cuda'):
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained CNN model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with test metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tracking variables
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    # No gradient computation for evaluation
    with torch.no_grad():
        for batch in test_loader:
            # Get data
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Track loss and predictions
            test_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = accuracy_score(test_labels, test_preds)
    
    # Generate detailed metrics
    class_report = classification_report(test_labels, test_preds, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, test_preds)
    
    # Compile results
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    # Print results
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/cnn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def visualize_attention(model, test_loader, device, n_samples=5):
    """Visualize which regions of sequences contribute most to classification."""
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

    # Define weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create model
    model = DNACNN(
        n_classes=n_classes,
        seq_length=window_size,
        n_channels=5,  # A, C, G, T, N
        conv_channels=[32, 64, 128],
        kernel_sizes=[11, 7, 3],
        fc_sizes=[512, 256],
        dropout=0.3
    )
    
    # Add multi-GPU support
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
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
        early_stopping=5,
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