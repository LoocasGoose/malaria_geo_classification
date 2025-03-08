'''
cnn_classifier_iterative.py
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
from torch.nn.modules import loss
import torch.optim as optim
import os
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.GenomicSequence import GenomicSequenceDataset

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

class AttentionPooling(nn.Module):
    """Simple attention mechanism to focus on important sequence regions
    
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

class DNACNN(nn.Module):
    """Version 3: Biologically-aware CNN Architecture
    
    Features:
    - Strand-symmetric convolutions for biological relevance
    - Attention mechanism to focus on important regions
    - Deeper architecture with three convolutional layers
    """

    def __init__(self, n_classes, seq_length=1000, n_channels=5):
        """Initialize the DNA CNN model."""
        
        super(DNACNN, self).__init__()

        # Strand-symmetric convolutional layers
        self.conv1 = StrandSymmetricConv(n_channels, 32, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = StrandSymmetricConv(32, 64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = StrandSymmetricConv(64, 128, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()  
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Attention pooling
        self.attention = AttentionPooling(128)

        # Calculate flattened size (will be different with attention)
        self.fc = nn.Linear(128, 128)  # Attention outputs [batch, channels]
        self.fc_relu = nn.ReLU()
        self.output = nn.Linear(128, n_classes)

    def forward(self, x):
        """Forward pass with strand-symmetric convolutions and attention."""

        x = x.permute(0, 2, 1)  
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Apply attention pooling instead of flatten
        x = self.attention(x)
        
        # Fully connected layers (no need for view/flatten)
        x = self.fc(x)
        x = self.fc_relu(x)
        x = self.output(x)

        return x
    
def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
          n_epochs=30, device='cpu', early_stopping_patience=5):
    """Train the DNA CNN model with early stopping and LR scheduling."""

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

    model = DNACNN(n_classes=n_classes, seq_length=1000, n_channels=5)
    logging.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer instead of SGD
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    try:
        # Train with early stopping and LR scheduling
        model, history = train(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=30,
            device=device,
            early_stopping_patience=5
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
        }, "models/cnn_v3.pt")
        logging.info(f"Model saved to models/cnn.pt")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

    return model, history

if __name__ == "__main__":
    main()
