'''
naive_bayes_classifier.py
~~~~~~~~~~

Purpose: 
    Train/evaluate the Multinomial Naive Bayes model with hyperparameter tuning.
Functions:
    train_nb(X_train, y_train) to fit the model.
    evaluate_nb(model, X_test, y_test) to compute metrics.
Output: 
    Saves the trained model and performance metrics.
'''

import os
import logging
import json
import time
from scipy.sparse import load_npz, csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(split_dir):
    """
    Load the split data from the specified directory.
    Ensures matrices are in CSR format for efficient computation.
    """
    def load_csr(file_path):
        matrix = load_npz(file_path)
        return matrix.tocsr() if not isinstance(matrix, csr_matrix) else matrix
    
    X_train = load_csr(os.path.join(split_dir, "train_features.npz"))
    X_val = load_csr(os.path.join(split_dir, "val_features.npz"))
    X_test = load_csr(os.path.join(split_dir, "test_features.npz"))
    
    # Try loading compressed labels first
    try:
        if os.path.exists(os.path.join(split_dir, "train_labels.npz")):
            y_train = np.load(os.path.join(split_dir, "train_labels.npz"))["labels"]
            y_val = np.load(os.path.join(split_dir, "val_labels.npz"))["labels"]
            y_test = np.load(os.path.join(split_dir, "test_labels.npz"))["labels"]
        else:
            y_train = np.load(os.path.join(split_dir, "train_labels.npy"))
            y_val = np.load(os.path.join(split_dir, "val_labels.npy"))
            y_test = np.load(os.path.join(split_dir, "test_labels.npy"))
    except Exception as e:
        logging.error(f"Error loading labels: {e}")
        raise
    
    logging.info(f"Loaded data: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_nb(X_train, y_train):
    """
    Train a Multinomial Naive Bayes model with feature selection and hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : sparse matrix
        Training feature matrix
    y_train : array
        Training labels
        
    Returns:
    --------
    model : Pipeline
        Trained model
    """
    logging.info("Training Naive Bayes model with feature selection...")
    
    # Create a pipeline with feature selection and classifier
    pipeline = Pipeline([
        ('feature_selector', SelectKBest(chi2)),
        ('classifier', MultinomialNB())
    ])
    
    # Define parameter grid
    param_dist = {
        'feature_selector__k': [5000, 10000, 20000, 'all'],
        'classifier__alpha': np.logspace(-3, 2, 20),  # Log-spaced values for smoothing
        'classifier__fit_prior': [True, False]
    }
    
    # Use randomized search for efficiency
    start_time = time.time()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,       # Cross-validation folds
        scoring='accuracy',
        random_state=42,
        n_jobs=-1   # Use all available cores
    )
    
    search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logging.info(f"Training completed in {training_time:.2f} seconds")
    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    
    # Get feature importance if feature selection was used
    best_model = search.best_estimator_
    if best_model.named_steps['feature_selector'].k != 'all':
        k_features = best_model.named_steps['feature_selector'].k
        logging.info(f"Selected {k_features} features out of {X_train.shape[1]}")
    
    return best_model

def evaluate_nb(model, X_test, y_test):
    """
    Evaluate the trained model and save detailed metrics.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model
    X_test : sparse matrix
        Test feature matrix
    y_test : array
        Test labels
    
    Returns:
    --------
    metrics : dict
        Performance metrics
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Log performance
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Inference time: {inference_time:.4f}s ({len(X_test)/inference_time:.1f} samples/sec)")
    
    # Detailed metrics
    metrics = {
        "accuracy": float(accuracy),
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
    Save the trained model using joblib for better handling of sparse matrices.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model
    output_path : str
        Path to save the model file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path, compress=3)
    logging.info(f"Model saved to {output_path} (size: {os.path.getsize(output_path)/1e6:.1f} MB)")

def main():
    # Define the directory where the split data is stored
    split_dir = "data/split"
    
    # Load the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(split_dir)
    
    # Train the model with hyperparameter tuning
    model = train_nb(X_train, y_train)
    
    # Evaluate on validation set
    logging.info("Evaluating on validation set:")
    evaluate_nb(model, X_val, y_val)
    
    # Evaluate on test set
    logging.info("Evaluating on test set:")
    test_metrics = evaluate_nb(model, X_test, y_test)
    
    # Save the model
    model_output_path = "models/nb_model.joblib"
    save_model(model, model_output_path)
    
    logging.info("Naive Bayes training and evaluation complete.")

if __name__ == "__main__":
    main()

