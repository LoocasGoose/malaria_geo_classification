'''
    cnn_standard.py
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim
import os
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.genomic_sequences import GenomicSequenceDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_reverse_complement(x):
    """Create reverse complement of DNA sequence tensor.
    
    Args:
        x: DNA tensor in one-hot format [batch, channels, seq_len]
            Channels: [A, C, G, T, N]
    
    Returns:
        Reverse complemented tensor with same shape
    """
    # Create a new tensor for reverse complement
    x_rc = torch.zeros_like(x)
    
    # Reverse the order of the sequence
    x_rc = torch.flip(x, dims=[2])
    
    # Swap complementary bases (A↔T, C↔G)
    x_rc[:, 0] = x[:, 3]  # A → T
    x_rc[:, 1] = x[:, 2]  # C → G
    x_rc[:, 2] = x[:, 1]  # G → C
    x_rc[:, 3] = x[:, 0]  # T → A
    x_rc[:, 4] = x[:, 4]  # N remains N
    
    return x_rc

class StrandSymmetricConv(nn.Module):
    """DNA strand-aware convolutional layer
    
    Processes DNA sequences in both forward and reverse-complement 
    directions simultaneously. This ensures the model detects patterns
    regardless of which DNA strand they appear on.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x):
        # Compute both forward and reverse-complement convolutions
        x_rc = create_reverse_complement(x)
        both_x = torch.cat([x, x_rc], dim=0)
        
        # Run convolution once on combined batch
        both_results = self.conv(both_x)
        
        # Split results back
        batch_size = x.size(0)
        x_fwd = both_results[:batch_size]
        x_rev = both_results[batch_size:]
        
        # Take maximum activation from either strand
        return torch.max(x_fwd, torch.flip(x_rev, dims=[2]))

class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input sequence.
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum length of the sequence
    """
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor [batch, channels, seq_len]
        Returns:
            Tensor with positional encoding added [batch, channels+d_model, seq_len]
        """
        # Get sequence length
        seq_len = x.size(2)
        batch_size = x.size(0)
        
        # Get positional encoding for this sequence length
        pos_enc = self.pe[:seq_len, :].transpose(0, 1).unsqueeze(0)
        
        # Expand to batch size
        pos_enc = pos_enc.expand(batch_size, -1, -1)
        
        # Concatenate along channel dimension
        return torch.cat([x, pos_enc], dim=1)

class AttentionPooling(nn.Module):
    """Attention mechanism to focus on important sequence regions
    
    DNA contains sparse functional elements - most of the sequence may be 
    less informative, while short motifs (e.g., binding sites) can be 
    critical. Attention learns to focus on these regions.
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
    """Residual block with strand-symmetric convolution
    
    Combines residual connections with strand symmetry for stable
    training of deeper networks.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
        super().__init__()
        self.conv = StrandSymmetricConv(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (if dimensions don't match)
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Main path
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Skip connection
        if self.skip is not None:
            residual = self.skip(x)
        
        # Add residual connection
        out += residual
        return self.relu(out)

class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss
    
    Adds regularization by preventing model from becoming too confident.
    """
    def __init__(self, smoothing=0.1, classes=10):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        # Convert to log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create one-hot encoding
        with torch.no_grad():
            target_one_hot = torch.zeros_like(pred)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        
        # Apply label smoothing
        target_smooth = target_one_hot * (1.0 - self.smoothing) + self.smoothing / self.classes
        
        # Calculate loss
        return -torch.sum(target_smooth * log_probs, dim=1).mean()

class DNACNN(nn.Module):
    """Standard CNN Architecture with Residual Connections
    
    Features:
    - Residual connections for gradient flow
    - Batch normalization for training stability
    - Positional encoding for genomic context
    - Dropout layers for regularization
    - Strand-symmetric convolutions for biological relevance
    - Attention mechanism to focus on important regions
    """

    def __init__(self, n_classes, seq_length=1000, n_channels=5, dropout=0.3):
        """Initialize the DNA CNN model."""
        
        super(DNACNN, self).__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=64, max_len=2000)
        
        # Input channels will be augmented with positional encoding
        augmented_channels = n_channels + 64
        
        # Residual convolution blocks with batch norm and dropout
        self.conv1 = ResidualStrandConv(augmented_channels, 32, kernel_size=5, dropout=dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = ResidualStrandConv(32, 64, kernel_size=5, dropout=dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = ResidualStrandConv(64, 128, kernel_size=5, dropout=dropout)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Attention pooling
        self.attention = AttentionPooling(128)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(128, 512)
        self.fc_relu1 = nn.ReLU()
        self.fc_dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 128)
        self.fc_relu2 = nn.ReLU()
        self.fc_dropout2 = nn.Dropout(dropout)
        
        self.output = nn.Linear(128, n_classes)

    def forward(self, x):
        """Forward pass with positional encoding, residual blocks, and attention."""

        # Rearrange for 1D convolution
        x = x.permute(0, 2, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # First conv block with residual connection
        x = self.conv1(x)
        x = self.pool1(x)

        # Second conv block with residual connection
        x = self.conv2(x)
        x = self.pool2(x)

        # Third conv block with residual connection
        x = self.conv3(x)
        x = self.pool3(x)

        # Apply attention pooling
        x = self.attention(x)
        
        # Fully connected layers with dropout
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc_dropout1(x)
        
        x = self.fc2(x)
        x = self.fc_relu2(x)
        x = self.fc_dropout2(x)
        
        x = self.output(x)

        return x
    
def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
          n_epochs=30, device='cpu', early_stopping_patience=5, grad_clip_val=1.0):
    """Train the DNA CNN model with gradient clipping and early stopping."""

    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)  # ReduceLROnPlateau needs validation loss

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                model.load_state_dict(best_model_state)
                break

        logging.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    return model, history

def evaluate(model, test_loader, criterion, device='cpu'):
    """Evaluate the DNA CNN model."""

    model.to(device)
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

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = accuracy_score(test_labels, test_preds)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return test_acc, test_loss
            

def main():
    """Main function to run the CNN training and evaluation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_dataset = GenomicSequenceDataset(
        split_dir="data/split",
        split_type="train",
        window_size=1000,
        stride=500,
        cache_size=128
    )

    val_dataset = GenomicSequenceDataset(
        split_dir="data/split",
        split_type="val",
        window_size=1000,
        stride=500,
        cache_size=128
    )

    test_dataset = GenomicSequenceDataset(
        split_dir="data/split",
        split_type="test",
        window_size=1000,
        stride=500,
        cache_size=128
    )

    batch_size = 128 if device.type == "cuda" else 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    n_classes = len(train_dataset.encoder.classes_)
    logging.info(f"Number of classes: {n_classes}")
    logging.info(f"Training samples: {len(train_dataset)}")

    model = DNACNN(n_classes=n_classes, seq_length=1000, n_channels=5, dropout=0.3)
    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Use label smoothing loss for regularization
    criterion = LabelSmoothingLoss(smoothing=0.1, classes=n_classes)
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    try:
        # Train with early stopping, LR scheduling, and gradient clipping
        model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=30,
            device=device,
            early_stopping_patience=5,
            grad_clip_val=1.0  # Add gradient clipping
        )
        
        # Evaluate on test set
        test_acc, test_loss = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'test_acc': test_acc,
            'test_loss': test_loss
        }, "models/cnn_standard.pt")
        logging.info(f"Model saved to models/cnn_standard.pt")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

    return model, history

if __name__ == "__main__":
    main()
