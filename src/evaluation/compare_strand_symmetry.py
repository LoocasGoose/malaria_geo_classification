import torch
import numpy as np
import time
import os
from torch import nn
from torch.utils.data import DataLoader
from src.models.cnn_advanced import DNACNN, StrandSymmetricConv, StandardConv
from src.data.genomic_sequences import GenomicSequenceDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def create_strand_symmetric_model(n_classes, seq_length=1000, symmetric=True):
    """Create model with or without symmetric convolutions.
    
    Args:
        n_classes: Number of output classes
        seq_length: Length of input sequences
        symmetric: Whether to use strand-symmetric convolutions
        
    Returns:
        DNACNN model with specified configuration
    """
    # Choose the appropriate convolution type
    conv_type = 'symmetric' if symmetric else 'standard'
    
    # Create the model
    model = DNACNN(
        n_classes=n_classes,
        seq_length=seq_length,
        n_channels=5,  # A, C, G, T, N
        conv_channels=[64, 128, 256],
        kernel_sizes=[15, 9, 5],
        fc_sizes=[1024, 512],
        dropout=0.4,
        conv_type=conv_type
    )
    
    return model

def evaluate_strand_symmetry_effect(test_loader, device='cuda', model_dir='models'):
    """Evaluate the effect of strand symmetry on model performance.
    
    Args:
        test_loader: DataLoader with test data
        device: Computing device
        model_dir: Directory where models are saved
        
    Returns:
        dict: Results for both models with and without strand symmetry
    """
    # Get number of classes from test loader
    sample_batch = next(iter(test_loader))
    n_classes = len(np.unique(sample_batch['label'].numpy()))
    seq_length = sample_batch['sequence'].shape[2]
    
    # Load or create models
    symmetric_model_path = os.path.join(model_dir, 'cnn_advanced_symmetric.pt')
    standard_model_path = os.path.join(model_dir, 'cnn_advanced_standard.pt')
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Initialize results
    results = {}
    
    # Test models if they exist
    for model_type, model_path in [
        ('Advanced CNN (Symmetric)', symmetric_model_path),
        ('Advanced CNN (Standard)', standard_model_path)
    ]:
        try:
            # Try to load the model
            model = create_strand_symmetric_model(
                n_classes=n_classes, 
                seq_length=seq_length,
                symmetric=(model_type == 'Advanced CNN (Symmetric)')
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            
            # Evaluate model
            print(f"Evaluating {model_type}...")
            model.eval()
            
            # Track metrics
            test_loss = 0.0
            all_preds = []
            all_labels = []
            inference_times = []
            
            with torch.no_grad():
                for batch in test_loader:
                    # Get data
                    sequences = batch['sequence'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(sequences)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Track loss and predictions
                    test_loss += loss.item() * sequences.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            test_loss = test_loss / len(test_loader.dataset)
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Get class report
            class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
            
            # Store results
            results[model_type] = {
                'model_name': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_loss': test_loss,
                'class_report': class_report,
                'predictions': all_preds,
                'true_labels': all_labels,
                'avg_inference_time': sum(inference_times) / len(inference_times),
                'inference_times': inference_times
            }
            
            print(f"{model_type} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"Could not evaluate {model_type}: {e}")
    
    return results

def plot_strand_symmetry_comparison(results):
    """Plot comparison of models with and without strand symmetry.
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if len(results) < 2:
        print("Not enough models to compare")
        return None
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    model_types = list(results.keys())
    
    # Extract metrics
    metric_values = {
        'Accuracy': [results[model]['accuracy'] for model in model_types],
        'Precision': [results[model]['precision'] for model in model_types],
        'Recall': [results[model]['recall'] for model in model_types],
        'F1 Score': [results[model]['f1_score'] for model in model_types],
        'Inference Time (ms)': [results[model]['avg_inference_time'] * 1000 for model in model_types]
    }
    
    # Calculate relative improvement
    rel_improvements = {}
    baseline = model_types[1]  # Using non-symmetric as baseline
    for metric in metrics:
        baseline_value = metric_values[metric][1]  # Non-symmetric value
        symmetric_value = metric_values[metric][0]  # Symmetric value
        rel_improvements[metric] = (symmetric_value - baseline_value) / baseline_value * 100
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot metrics
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, [metric_values[m][0] for m in metrics], width, label=model_types[0])
    ax1.bar(x + width/2, [metric_values[m][1] for m in metrics], width, label=model_types[1])
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison: Strand Symmetry Effect')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # Add value labels
    for i, v in enumerate([metric_values[m][0] for m in metrics]):
        ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate([metric_values[m][1] for m in metrics]):
        ax1.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    # Plot relative improvements
    ax2.bar(metrics, [rel_improvements[m] for m in metrics], color='green')
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Relative Improvement (%)')
    ax2.set_title('Relative Improvement with Strand Symmetry')
    
    # Add value labels
    for i, v in enumerate([rel_improvements[m] for m in metrics]):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    return fig

def run_strand_symmetry_analysis(test_loader, device='cuda', output_dir='reports/figures'):
    """Run complete strand symmetry analysis and save results.
    
    Args:
        test_loader: DataLoader with test data
        device: Computing device
        output_dir: Directory to save output figures
        
    Returns:
        dict: Evaluation results for each model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate models
    results = evaluate_strand_symmetry_effect(test_loader, device)
    
    # Plot comparison if we have enough models
    if len(results) >= 2:
        fig = plot_strand_symmetry_comparison(results)
        if fig:
            fig.savefig(os.path.join(output_dir, 'strand_symmetry_comparison.png'), dpi=300)
            plt.close(fig)
    
    return results

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
    results = run_strand_symmetry_analysis(DataLoader(GenomicSequenceDataset(
        split_dir=config["data_dir"],
        split_type="test",
        window_size=config["seq_length"],
        stride=config["stride"]
    ), batch_size=config["batch_size"]), config["device"])
    
    # Save results
    torch.save(results, "symmetry_comparison_results.pt")
    print("Comparison complete! Results saved to symmetry_comparison_results.pt") 