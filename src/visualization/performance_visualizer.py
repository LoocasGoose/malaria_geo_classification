import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', figsize=(10, 8)):
    """Plot confusion matrix with percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Title of the plot
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'{title} (Counts)')
    
    # Plot percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title(f'{title} (Percentages)')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(roc_data, figsize=(10, 8)):
    """Plot ROC curves for multiple models.
    
    Args:
        roc_data: Dictionary containing ROC curve data for each model
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, data in roc_data.items():
        # Calculate micro-average ROC curve
        all_fpr = np.unique(np.concatenate([data['fpr'][i] for i in range(data['n_classes'])]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(data['n_classes']):
            mean_tpr += np.interp(all_fpr, data['fpr'][i], data['tpr'][i])
        
        mean_tpr /= data['n_classes']
        roc_auc = np.mean(list(data['roc_auc'].values()))
        
        # Plot the curve
        ax.plot(all_fpr, mean_tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Add diagonal line for random performance
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison(comparison_df, figsize=(12, 10)):
    """Plot comparison of key metrics across models.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        sns.barplot(x='Model', y=metric, data=comparison_df, ax=ax)
        ax.set_title(f'Comparison of {metric}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_complexity_tradeoff(tradeoff_results, figsize=(10, 8)):
    """Plot the complexity vs. performance tradeoff.
    
    Args:
        tradeoff_results: Dictionary with tradeoff analysis results
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy vs. Inference Time
    ax1.scatter(tradeoff_results['inference_times'], tradeoff_results['accuracies'], s=100)
    for i, model in enumerate(tradeoff_results['models']):
        ax1.annotate(model, 
                   (tradeoff_results['inference_times'][i], tradeoff_results['accuracies'][i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs. Inference Time')
    
    # Efficiency Score
    ax2.bar(tradeoff_results['models'], tradeoff_results['efficiency_scores'])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Efficiency Score (Accuracy per ms)')
    ax2.set_title('Model Efficiency Comparison')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_per_class_performance(per_class_results, metric='F1 Score', figsize=(12, 8)):
    """Plot per-class performance across models.
    
    Args:
        per_class_results: DataFrame with per-class performance metrics
        metric: Metric to plot ('F1 Score', 'Precision', or 'Recall')
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(x='Class', y=metric, hue='Model', data=per_class_results, ax=ax)
    ax.set_title(f'Per-Class {metric} Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_training_history(history, figsize=(12, 5)):
    """Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary containing training history
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_model_architecture_comparison(model_sizes, param_counts, figsize=(10, 5)):
    """Plot model architecture comparison.
    
    Args:
        model_sizes: Dictionary mapping model names to sizes in MB
        param_counts: Dictionary mapping model names to parameter counts
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    models = list(model_sizes.keys())
    sizes = list(model_sizes.values())
    params = list(param_counts.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot model sizes
    ax1.bar(models, sizes)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Model Size (MB)')
    ax1.set_title('Model Size Comparison')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot parameter counts
    ax2.bar(models, params)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Parameter Count (millions)')
    ax2.set_title('Parameter Count Comparison')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for ax, values in zip([ax1, ax2], [sizes, params]):
        for i, v in enumerate(values):
            unit = 'MB' if ax == ax1 else 'M'
            display_val = v if ax == ax1 else v / 1e6
            ax.text(i, v, f'{display_val:.2f} {unit}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_challenging_classes(per_class_results, challenging_classes, metric='F1 Score', figsize=(12, 6)):
    """Plot performance for challenging classes.
    
    Args:
        per_class_results: DataFrame with per-class performance metrics
        challenging_classes: List of challenging class names
        metric: Metric to plot ('F1 Score', 'Precision', or 'Recall')
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Filter for challenging classes
    data = per_class_results[per_class_results['Class'].isin(challenging_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(x='Class', y=metric, hue='Model', data=data, ax=ax)
    ax.set_title(f'Performance on Challenging Classes ({metric})')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig 