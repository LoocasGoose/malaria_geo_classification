import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from src.models.cnn_classifier_ultimate import (
    DNACNN,
    StrandSymmetricConv,
    GenomicSequenceDataset,
    train,
    evaluate,
    create_reverse_complement
)
import matplotlib.pyplot as plt

def create_model(use_symmetric, n_classes, seq_length=1000):
    """Create model with or without symmetric convolutions"""
    conv_layer = StrandSymmetricConv if use_symmetric else nn.Conv1d
    return DNACNN(
        n_classes=n_classes,
        seq_length=seq_length,
        conv_channels=[64, 128, 256],
        kernel_sizes=[15, 9, 5],
        conv_layer=conv_layer
    )

def train_comparison(config):
    """Run comparative training experiment"""
    # Initialize models
    models = {
        "Symmetric": create_model(True, config["n_classes"], config["seq_length"]),
        "Standard": create_model(False, config["n_classes"], config["seq_length"])
    }
    
    # Load dataset
    train_dataset = GenomicSequenceDataset(
        split_dir=config["data_dir"],
        split_type="train",
        window_size=config["seq_length"],
        stride=config["stride"]
    )
    val_dataset = GenomicSequenceDataset(
        split_dir=config["data_dir"],
        split_type="val",
        window_size=config["seq_length"],
        stride=config["stride"]
    )
    
    # Training results storage
    results = {}
    
    for model_name in models:
        print(f"\nTraining {model_name} model...")
        model = models[model_name].to(config["device"])
        
        # Shared training parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train model
        trained_model, history = train(
            model=model,
            train_loader=DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True),
            val_loader=DataLoader(val_dataset, batch_size=config["batch_size"]),
            criterion=criterion,
            optimizer=optimizer,
            n_epochs=config["epochs"],
            device=config["device"],
            early_stopping_patience=5
        )
        
        # Store results
        results[model_name] = {
            "history": history,
            "model": trained_model
        }
    
    return results

def plot_results(results):
    """Plot comparative training curves"""
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for model_name in results:
        plt.plot(results[model_name]["history"]["val_acc"], 
                label=f"{model_name} Val Acc")
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    for model_name in results:
        plt.plot(results[model_name]["history"]["val_loss"], 
                label=f"{model_name} Val Loss")
    plt.title("Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("strand_symmetry_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Configuration
    config = {
        "data_dir": "data/split",
        "n_classes": 25,
        "seq_length": 1000,
        "stride": 500,
        "batch_size": 64,
        "epochs": 30,
        "lr": 0.001,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(42)
    
    # Run comparison
    results = train_comparison(config)
    
    # Save and plot results
    torch.save(results, "symmetry_comparison_results.pt")
    plot_results(results)
    print("Comparison complete! Results saved to symmetry_comparison_results.pt") 