"""
Multinomial Naive Bayes Classifier for Malaria Geographic Origin Classification.

This module implements a Multinomial Naive Bayes approach for classifying malaria
parasite samples by geographic origin based on genomic variants. Key features include:

1. Feature selection using chi-squared test to reduce dimensionality
2. Hyperparameter tuning via randomized search cross-validation
3. Efficient handling of sparse genomic feature matrices
4. Comprehensive performance evaluation and metrics reporting
5. Model persistence with optimized storage

Design choices:
- Multinomial Naive Bayes is well-suited for sparse count/frequency data like genomic variants
- Feature selection reduces dimensionality and improves model interpretability
- Hyperparameter tuning optimizes smoothing and feature count for best performance
- Pipeline architecture ensures consistent preprocessing during training and inference
- Sparse matrix handling optimizes memory usage for large genomic datasets

Benefits over more complex models:
- Fast training and inference, even on large datasets
- Interpretable probabilistic outputs
- Robust performance with limited samples per class
- Low computational resource requirements
"""

import os
import logging
import json
import time
from scipy.sparse import load_npz, csr_matrix, issparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import joblib
import pickle
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier for genomic variant-based classification.
    
    This class wraps scikit-learn's MultinomialNB with additional functionality:
    1. Feature selection to improve performance on high-dimensional genomic data
    2. Hyperparameter optimization to find optimal smoothing parameters
    3. Efficient handling of sparse genomic feature matrices
    4. Methods for model persistence and loading
    
    The model is structured as a scikit-learn Pipeline with:
    - Feature selection using chi-squared test (SelectKBest)
    - Multinomial Naive Bayes classifier with tuned alpha (smoothing)
    
    Design choices:
    - Pipeline architecture ensures consistent preprocessing in training and inference
    - Sparse matrix handling enables efficient processing of large genomic datasets
    - Class-based implementation provides a consistent API with other models
    """
    
    def __init__(self):
        """
        Initialize the Multinomial Naive Bayes model.
        
        The model is initially empty and will be populated during training
        or when loading a pre-trained model.
        """
        self.model = None
        self.metrics = {}
        
    def train(self, X_train, y_train, param_grid=None, n_iter=20, cv=3):
        """
        Train the model with hyperparameter tuning.
        
        Args:
            X_train (scipy.sparse.csr_matrix): Training feature matrix
            y_train (numpy.ndarray): Training labels
            param_grid (dict, optional): Hyperparameter grid for tuning
            n_iter (int): Number of parameter settings to try
            cv (int): Number of cross-validation folds
            
        Returns:
            self: Trained model instance
        """
        self.model = train_nb(X_train, y_train, param_grid, n_iter, cv)
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (scipy.sparse.csr_matrix): Feature matrix
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        
        Args:
            X (scipy.sparse.csr_matrix): Feature matrix
            
        Returns:
            numpy.ndarray: Probability estimates for each class
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (scipy.sparse.csr_matrix): Test feature matrix
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Performance metrics
        """
        self.metrics = evaluate_nb(self.model, X_test, y_test)
        return self.metrics
    
    def save(self, output_path):
        """
        Save the trained model to disk.
        
        Args:
            output_path (str): Path to save the model file
            
        Returns:
            None
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        save_model(self.model, output_path)
    
    @classmethod
    def load(cls, input_path):
        """
        Load a pre-trained model from disk.
        
        Args:
            input_path (str): Path to the saved model file
            
        Returns:
            MultinomialNaiveBayes: Loaded model instance
        """
        instance = cls()
        instance.model = joblib.load(input_path)
        logging.info(f"Model loaded from {input_path}")
        return instance

def load_data(split_dir):
    """
    Load the split data from disk for model training and evaluation.
    
    This function loads feature matrices and labels for training, validation, and test sets
    from the specified directory. It ensures matrices are in CSR format for efficient
    sparse matrix operations, which is critical for memory-efficient processing of 
    high-dimensional genomic data.
    
    Args:
        split_dir (str): Directory containing split data files
            
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
            - X_train, X_val, X_test: Feature matrices (scipy.sparse.csr_matrix)
            - y_train, y_val, y_test: Label arrays (numpy.ndarray)
            
    Design choice:
        CSR format is used for sparse matrices as it provides efficient row slicing and
        matrix-vector operations, which are common in Naive Bayes training and inference.
        Both .npz and .npy formats are supported for labels for compatibility.
    """
    def load_csr(file_path):
        """Load matrix and ensure it's in CSR format for efficient computation."""
        matrix = load_npz(file_path)
        return matrix.tocsr() if not isinstance(matrix, csr_matrix) else matrix
    
    # Attempt to load features from standardized file names (X_train.npz, etc.)
    # Fall back to legacy names (train_features.npz, etc.) if needed
    try:
        if os.path.exists(os.path.join(split_dir, "X_train.npz")):
            X_train = load_csr(os.path.join(split_dir, "X_train.npz"))
            X_val = load_csr(os.path.join(split_dir, "X_val.npz"))
            X_test = load_csr(os.path.join(split_dir, "X_test.npz"))
        else:
            X_train = load_csr(os.path.join(split_dir, "train_features.npz"))
            X_val = load_csr(os.path.join(split_dir, "val_features.npz"))
            X_test = load_csr(os.path.join(split_dir, "test_features.npz"))
    except Exception as e:
        logging.error(f"Error loading feature matrices: {e}")
        raise
    
    # Try loading compressed labels first, then fall back to .npy
    # Now with safer label handling for different .npz structures
    try:
        # Train labels
        if os.path.exists(os.path.join(split_dir, "train_labels.npz")):
            npz_file = np.load(os.path.join(split_dir, "train_labels.npz"))
            # Try different possible keys - labels is primary, but fall back to arr_0 or first key
            if "labels" in npz_file:
                y_train = npz_file["labels"]
            elif "arr_0" in npz_file:
                y_train = npz_file["arr_0"]
            else:
                y_train = npz_file[list(npz_file.keys())[0]]
        else:
            y_train = np.load(os.path.join(split_dir, "train_labels.npy"))
        
        # Validation labels
        if os.path.exists(os.path.join(split_dir, "val_labels.npz")):
            npz_file = np.load(os.path.join(split_dir, "val_labels.npz"))
            if "labels" in npz_file:
                y_val = npz_file["labels"]
            elif "arr_0" in npz_file:
                y_val = npz_file["arr_0"]
            else:
                y_val = npz_file[list(npz_file.keys())[0]]
        else:
            y_val = np.load(os.path.join(split_dir, "val_labels.npy"))
        
        # Test labels
        if os.path.exists(os.path.join(split_dir, "test_labels.npz")):
            npz_file = np.load(os.path.join(split_dir, "test_labels.npz"))
            if "labels" in npz_file:
                y_test = npz_file["labels"]
            elif "arr_0" in npz_file:
                y_test = npz_file["arr_0"]
            else:
                y_test = npz_file[list(npz_file.keys())[0]]
        else:
            y_test = np.load(os.path.join(split_dir, "test_labels.npy"))
    except Exception as e:
        logging.error(f"Error loading labels: {e}")
        raise
    
    logging.info(f"Loaded data: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    # Validate data for non-negative values (chi2 requirement)
    if issparse(X_train):
        if X_train.data.min() < 0:
            warnings.warn("X_train contains negative values which may cause issues with chi² feature selection.")
    elif np.min(X_train) < 0:
        warnings.warn("X_train contains negative values which may cause issues with chi² feature selection.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_nb(X_train, y_train, param_grid=None, n_iter=20, cv=3):
    """
    Train a Multinomial Naive Bayes model with feature selection and hyperparameter tuning.
    
    This function creates a scikit-learn Pipeline with feature selection (chi-squared)
    followed by a Multinomial Naive Bayes classifier. It performs randomized search
    cross-validation to find optimal hyperparameters, focusing on:
    1. Number of features to select (k)
    2. Smoothing parameter (alpha)
    3. Whether to use class prior probabilities (fit_prior)
    
    Args:
        X_train (scipy.sparse.csr_matrix): Training feature matrix
        y_train (numpy.ndarray): Training labels
        param_grid (dict, optional): Hyperparameter grid for tuning
                                  If None, default grid is used
        n_iter (int): Number of parameter settings to try
        cv (int): Number of cross-validation folds
        
    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline
        
    Design choices:
        - Feature selection reduces dimensionality to improve model performance
          and efficiency, which is important for genomic data with hundreds of
          thousands of features
        - Randomized search is more efficient than grid search for high-dimensional
          parameter spaces
        - Log-spaced alpha values explore a wide range of smoothing strengths
        - Pipeline architecture ensures preprocessing steps are applied consistently
          during training and prediction
    """
    logging.info("Training Multinomial Naive Bayes model with feature selection...")
    
    # Validate input data for chi2 feature selection
    if issparse(X_train):
        if X_train.data.min() < 0:
            raise ValueError("X_train contains negative values. chi² feature selection requires non-negative inputs.")
    elif np.min(X_train) < 0:
        raise ValueError("X_train contains negative values. chi² feature selection requires non-negative inputs.")
    
    # Check for class imbalance
    class_counts = np.bincount(y_train)
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 3:
        logging.warning(f"Dataset has significant class imbalance (ratio: {imbalance_ratio:.1f}). "
                      f"Using weighted metrics for optimization.")
        scoring = 'f1_weighted'
    else:
        scoring = 'accuracy'
    
    # Create a pipeline with feature selection and classifier
    pipeline = Pipeline([
        ('feature_selector', SelectKBest(chi2)),
        ('classifier', MultinomialNB())
    ])
    
    # Define default parameter grid if none provided
    if param_grid is None:
        # Dynamic k values based on dataset size
        max_features = X_train.shape[1]
        k_values = []
        for k in [2000, 5000, 10000, 20000]:
            if k < max_features:
                k_values.append(k)
        if not k_values:  # If all k values are too large
            k_values = [max(100, max_features // 10), max(200, max_features // 5)]
        k_values.append('all')
        
        param_grid = {
            'feature_selector__k': k_values,
            'classifier__alpha': np.logspace(-3, 2, 20),  # Log-spaced values for smoothing
            'classifier__fit_prior': [True, False]
        }
    
    # Use randomized search for efficiency
    start_time = time.time()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,  # Number of parameter settings sampled
        cv=cv,          # Cross-validation folds
        scoring=scoring,  # Use appropriate scoring metric based on class balance
        random_state=42,
        n_jobs=-1       # Use all available cores
    )
    
    search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logging.info(f"Training completed in {training_time:.2f} seconds")
    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Best cross-validation {scoring}: {search.best_score_:.4f}")
    
    # Get feature importance if feature selection was used
    best_model = search.best_estimator_
    if best_model.named_steps['feature_selector'].k != 'all':
        k_features = best_model.named_steps['feature_selector'].k
        logging.info(f"Selected {k_features} features out of {X_train.shape[1]}")
    
    return best_model

def evaluate_nb(model, X_test, y_test):
    """
    Evaluate the trained model and calculate detailed performance metrics.
    
    This function evaluates a trained Naive Bayes model on test data,
    calculating a comprehensive set of performance metrics including:
    - Accuracy
    - Per-class precision, recall, and F1 score
    - Confusion matrix
    - Inference speed (samples per second)
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained model
        X_test (scipy.sparse.csr_matrix): Test feature matrix
        y_test (numpy.ndarray): Test labels
    
    Returns:
        dict: Performance metrics dictionary containing:
            - accuracy: Overall classification accuracy
            - classification_report: Per-class precision, recall, F1
            - confusion_matrix: Matrix of actual vs. predicted classes
            - inference_time: Total time for prediction
            - samples_per_second: Throughput rate
            
    Design choice:
        Comprehensive evaluation metrics provide insights into model performance
        across different geographic regions, which is crucial for understanding
        the practical utility of the model in real-world applications where some
        regions may be more important or challenging than others.
    """
    # Validate input data for chi2 feature selection - prevent runtime errors
    if 'feature_selector' in model.named_steps and isinstance(model.named_steps['feature_selector'], SelectKBest):
        if issparse(X_test):
            if X_test.data.min() < 0:
                raise ValueError("X_test contains negative values. chi² feature selection requires non-negative inputs.")
        elif np.min(X_test) < 0:
            raise ValueError("X_test contains negative values. chi² feature selection requires non-negative inputs.")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate class-weighted metrics for better assessment of imbalanced datasets
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']
    
    # Log performance
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Weighted F1-score: {weighted_f1:.4f}")
    logging.info(f"Inference time: {inference_time:.4f}s ({len(X_test)/inference_time:.1f} samples/sec)")
    
    # Check for class-specific issues
    class_f1_scores = {k: v['f1-score'] for k, v in report.items() 
                      if isinstance(k, (int, str)) and k not in ['accuracy', 'macro avg', 'weighted avg']}
    if min(class_f1_scores.values()) < 0.5:
        worst_class = min(class_f1_scores.items(), key=lambda x: x[1])
        logging.warning(f"Poor performance on class {worst_class[0]}: F1={worst_class[1]:.2f}")
    
    # Detailed metrics
    metrics = {
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "inference_time": float(inference_time),
        "samples_per_second": float(len(X_test)/inference_time)
    }
    
    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/nb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def save_model(model, output_path):
    """
    Save the trained model to disk using joblib serialization.
    
    This function saves a trained model to disk in an efficient format that
    preserves the sparse matrix structure of internal components. Joblib is
    used rather than pickle for better handling of sparse matrices and
    compression to reduce file size.
    
    Args:
        model (sklearn.pipeline.Pipeline): Trained model pipeline
        output_path (str): Path where model should be saved
            
    Returns:
        None: Model is saved to disk at specified location
            
    Design choice:
        Joblib serialization is used with compression level 3, which provides
        a good balance between file size and loading speed. This is important
        for genomic models which may contain large sparse matrices.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    logging.info(f"Model saved to {output_path} (size: {os.path.getsize(output_path)/1e6:.1f} MB)")

def main():
    """
    Main function to train, evaluate, and save a Naive Bayes classifier.
    
    This function orchestrates the end-to-end workflow for training a
    Multinomial Naive Bayes classifier for geographic origin prediction:
    1. Load split data from disk
    2. Train model with hyperparameter tuning
    3. Evaluate on test set 
    4. Save trained model to disk
    
    The function follows a modular design that separates data loading,
    model training, evaluation, and persistence into distinct steps.
    
    Returns:
        None: Results are saved to disk
            
    Design choices:
        - Integrated cross-validation during hyperparameter tuning replaces
          separate validation evaluation for more efficient data usage
        - Consistent directory structure matches other model types
        - Comprehensive logging provides transparency and debugging information
    """
    # Define the directory where the split data is stored
    split_dir = "data/split"
    
    # Load the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(split_dir)
    
    # Optional: Combine training and validation sets for final model training
    # This improves model performance by using more data after hyperparameter tuning
    # Uncomment to use
    # from scipy.sparse import vstack
    # X_train_full = vstack([X_train, X_val])
    # y_train_full = np.concatenate([y_train, y_val])
    # logging.info(f"Combined training and validation data: {X_train_full.shape}")
    
    # Train the model with hyperparameter tuning
    model = train_nb(X_train, y_train)
    
    # Evaluate directly on test set - validation already used in cross-validation
    logging.info("Evaluating on test set:")
    test_metrics = evaluate_nb(model, X_test, y_test)
    
    # Save the model
    model_output_path = "models/nb_model.joblib"
    save_model(model, model_output_path)
    
    logging.info("Naive Bayes training and evaluation complete.")

if __name__ == "__main__":
    main()

