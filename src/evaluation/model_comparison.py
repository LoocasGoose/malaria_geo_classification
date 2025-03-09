import numpy as np
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def compare_models(model_results_list):
    """Compare multiple models based on their evaluation results.
    
    Args:
        model_results_list: List of dictionaries containing model evaluation results
        
    Returns:
        dict: Comparison metrics and summary
    """
    # Extract basic metrics for all models
    comparison_df = pd.DataFrame([
        {
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1_score'],
            'Loss': result['test_loss'],
            'Inference Time (ms)': result['avg_inference_time'] * 1000  # Convert to ms
        }
        for result in model_results_list
    ])
    
    # Calculate relative improvements
    baseline = comparison_df.iloc[0]  # Assuming first model is baseline
    for i in range(1, len(comparison_df)):
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            rel_improvement = (comparison_df.iloc[i][metric] - baseline[metric]) / baseline[metric] * 100
            comparison_df.loc[i, f'{metric} Improvement (%)'] = rel_improvement
    
    # Compile results
    comparison_results = {
        'metrics_table': comparison_df,
        'model_results': model_results_list
    }
    
    return comparison_results

def calculate_roc_curves(model_results_list, classes=None):
    """Calculate ROC curves for multiple models.
    
    Args:
        model_results_list: List of dictionaries containing model evaluation results
        classes: List of class names
        
    Returns:
        dict: ROC curve data for each model
    """
    roc_data = {}
    
    for result in model_results_list:
        model_name = result['model_name']
        y_true = result['true_labels']
        
        # If probabilities are available, use them
        if 'probabilities' in result:
            y_score = result['probabilities']
        else:
            # One-hot encode predictions as a basic fallback
            n_classes = len(np.unique(y_true))
            y_score = label_binarize(result['predictions'], classes=range(n_classes))
        
        # One-hot encode true labels
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Store results
        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'n_classes': n_classes
        }
    
    return roc_data

def analyze_complexity_tradeoff(model_results_list):
    """Analyze the complexity vs. performance tradeoff.
    
    Args:
        model_results_list: List of dictionaries containing model evaluation results
        
    Returns:
        dict: Tradeoff analysis results
    """
    # Extract relevant metrics
    models = [r['model_name'] for r in model_results_list]
    accuracies = [r['accuracy'] for r in model_results_list]
    inference_times = [r['avg_inference_time'] * 1000 for r in model_results_list]  # Convert to ms
    
    # Calculate efficiency score (accuracy per ms)
    efficiency_scores = [acc / time for acc, time in zip(accuracies, inference_times)]
    
    # Compile results
    tradeoff_results = {
        'models': models,
        'accuracies': accuracies,
        'inference_times': inference_times,
        'efficiency_scores': efficiency_scores
    }
    
    return tradeoff_results

def analyze_per_class_performance(model_results_list, class_names=None):
    """Analyze per-class performance across models.
    
    Args:
        model_results_list: List of dictionaries containing model evaluation results
        class_names: List of class names
        
    Returns:
        pd.DataFrame: Per-class performance metrics across models
    """
    per_class_results = pd.DataFrame()
    
    for result in model_results_list:
        model_name = result['model_name']
        class_report = result['class_report']
        
        for class_idx, metrics in class_report.items():
            if class_idx in ['accuracy', 'macro avg', 'weighted avg']:
                continue
                
            if class_names is not None and int(class_idx) < len(class_names):
                class_label = class_names[int(class_idx)]
            else:
                class_label = f"Class {class_idx}"
                
            per_class_results = pd.concat([per_class_results, pd.DataFrame([{
                'Model': model_name,
                'Class': class_label,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1-score'],
                'Support': metrics['support']
            }])], ignore_index=True)
    
    return per_class_results

def identify_challenging_classes(per_class_results, threshold=0.7):
    """Identify classes that are challenging across all models.
    
    Args:
        per_class_results: DataFrame with per-class performance metrics
        threshold: F1 score threshold below which a class is considered challenging
        
    Returns:
        list: Names of challenging classes
    """
    # Get average F1 score for each class across all models
    class_avg_f1 = per_class_results.groupby('Class')['F1 Score'].mean()
    
    # Identify classes below threshold
    challenging_classes = class_avg_f1[class_avg_f1 < threshold].index.tolist()
    
    return challenging_classes

def identify_best_model_per_class(per_class_results):
    """Identify which model performs best for each class.
    
    Args:
        per_class_results: DataFrame with per-class performance metrics
        
    Returns:
        pd.DataFrame: Best model for each class
    """
    # Find model with highest F1 score for each class
    best_models = per_class_results.loc[per_class_results.groupby('Class')['F1 Score'].idxmax()]
    best_models = best_models[['Class', 'Model', 'F1 Score']]
    best_models = best_models.rename(columns={'F1 Score': 'Best F1 Score'})
    
    return best_models

def calculate_model_size(model):
    """Calculate the size of a PyTorch model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb 