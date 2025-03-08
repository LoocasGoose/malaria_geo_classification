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
import torch.optim as optim
import os
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

from data.GenomicSequence import GenomicSequenceDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DNACNN(nn.Module):
    """Basic Malaria DNA Analysis CNN Architecture"""

    def __init__(self, n_classes, seq_length=1000, n_channels=5):
        """Initialize the DNA CNN model. """
        
        super(DNACNN, self).__init__()

        self.conv = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        flatten_size = 32 * (seq_length // 2) * 32
        self.fc = nn.Linear(flatten_size, 128)
        self.fc_relu = nn.ReLU()
        self.output = nn.Linear(128, n_classes)

    def forward(self, x):
        """Forward pass of the DNA CNN model."""

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc_relu(x)
        x = self.output(x)

        return x
    
def train(model, train_loader, val_loader, criterion, optimizer, n_epochs=20, device='cpu'):
    """Train the DNA CNN model."""

    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
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
            _, predicted = outputs.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)

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
                _, predicted = outputs.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logging.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return model, history   

def evaluate(model, test_loader, device='cpu'):
    """Evaluate the DNA CNN model."""

    model.to(device)
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            outputs = model(sequences)
            _, predicted = outputs.max(outputs, 1)
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_classes = len(train_dataset.encoder.classes_)

    model = DNACNN(n_classes=n_classes, seq_length=1000, n_channels=5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model, history = train(model, train_loader, val_loader, criterion, optimizer, n_epochs=20, device=device)

    test_acc, test_loss = evaluate(model, test_loader, criterion, device=device)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_v1.pt")
    logging.info(f"Model saved to models/cnn_v1.pt")

    return model, history

if __name__ == "__main__":
    main()
