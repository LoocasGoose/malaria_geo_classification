'''
data_splitter.py
~~~~~~~~~~
'''
import os
import pandas as pd
import pickle
import json
import hashlib
import logging
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load the preprocessed data from disk.
    
    Returns:
    --------
    metadata : DataFrame
        Sample metadata
    X : sparse matrix
        Feature matrix
    y : array
        Encoded geographic labels
    sample_ids : array
        Sample IDs
    encoder : LabelEncoder
        Label encoder for geographic labels
    """
    metadata = pd.read_csv("data/processed/filtered_metadata.csv")
    sample_ids = metadata["Sample"].values
    
    # Set Sample as index for faster lookups
    metadata.set_index("Sample", inplace=True, drop=False)
    y = metadata["encoded_country"].values
    
    # Load sparse matrix with validation
    X = load_npz("data/processed/variant_features.npz")
    if X.shape[0] != len(metadata):
        raise ValueError(f"Mismatch between features ({X.shape[0]}) and metadata ({len(metadata)})")
    
    with open("data/processed/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    
    return metadata, X, y, sample_ids, encoder

def split_data(features, labels, sample_ids, test_size=0.15, val_size=0.15, random_state=42, stratify=True):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    -----------
    features : array-like
        Feature matrix (k-mer frequencies)
    labels : array-like
        Geographic labels for each sample
    sample_ids : array-like
        Sample IDs for each sample
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of data for validation
    random_state : int
        Seed for reproducibility
    stratify : bool
        Whether to use stratified sampling
        
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test : arrays
        Split datasets and corresponding indices
    """
    # Calculate absolute sizes for more precise split control
    n_total = features.shape[0]
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    
    # First split: Separate out the test set
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        features, labels, sample_ids,
        test_size=n_test,
        stratify=labels if stratify else None,
        random_state=random_state
    )

    # Second split: Separate validation set from the remaining training data
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val,
        test_size=n_val,
        stratify=y_train_val if stratify else None,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test

def check_split_feasibility(labels, test_size, val_size, min_samples_per_class=5):
    """
    Check if the requested split is possible given the geographic distribution.
    Some regions might have too few samples to be represented in all splits.
    
    Parameters:
    -----------
    labels : array-like
        Geographic labels for each sample
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of data for validation
    min_samples_per_class : int
        Minimum number of samples required per class in each split
        
    Returns:
    --------
    bool
        True if the split is feasible, raises ValueError otherwise
    """
    # Simulate actual splits to check feasibility
    try:
        # First check test split
        _, y_test = train_test_split(
            labels, 
            test_size=test_size, 
            stratify=labels,
            random_state=0
        )
        
        # Then check validation split
        remaining_y = np.delete(labels, np.arange(len(y_test)))
        _, y_val = train_test_split(
            remaining_y,
            test_size=val_size/(1-test_size),
            stratify=remaining_y,
            random_state=0
        )
        
        # Check minimums in each split
        for split_name, split_labels in [("Training", remaining_y[len(y_val):]), 
                                         ("Validation", y_val), 
                                         ("Test", y_test)]:
            class_counts = Counter(split_labels)
            min_class = min(class_counts.items(), key=lambda x: x[1])
            if min_class[1] < min_samples_per_class:
                raise ValueError(f"{split_name} split would have class {min_class[0]} with only {min_class[1]} samples")
    
    except ValueError as e:
        raise ValueError(f"Split feasibility check failed: {str(e)}")
    
    # Print statistics about class distribution
    min_class = min(Counter(labels).items(), key=lambda x: x[1])
    max_class = max(Counter(labels).items(), key=lambda x: x[1])
    
    logging.info("Class distribution check passed:")
    logging.info(f"  Minimum class: {min_class[0]} with {min_class[1]} samples")
    logging.info(f"  Maximum class: {max_class[0]} with {max_class[1]} samples")
    logging.info(f"  Each split will have at least {min_samples_per_class} samples per class")
    
    return True

def save_splits(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, 
               idx_train=None, idx_val=None, idx_test=None, metadata=None, encoder=None):
    """
    Save the split datasets to disk for future use.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the splits
    X_train, X_val, X_test : sparse matrices
        Feature matrices for each split
    y_train, y_val, y_test : arrays
        Labels for each split
    idx_train, idx_val, idx_test : arrays, optional
        Sample IDs for each split
    metadata : DataFrame, optional
        Full metadata DataFrame
    encoder : LabelEncoder, optional
        Label encoder for countries
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature matrices
    save_npz(os.path.join(output_dir, "train_features.npz"), X_train)
    save_npz(os.path.join(output_dir, "val_features.npz"), X_val)
    save_npz(os.path.join(output_dir, "test_features.npz"), X_test)
    
    # Save labels
    np.save(os.path.join(output_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(output_dir, "val_labels.npy"), y_val)
    np.save(os.path.join(output_dir, "test_labels.npy"), y_test)
    
    # Save metadata if provided
    if metadata is not None and all(x is not None for x in [idx_train, idx_val, idx_test]):
        # More efficient index-based selection
        train_metadata = metadata.loc[idx_train].copy()
        val_metadata = metadata.loc[idx_val].copy()
        test_metadata = metadata.loc[idx_test].copy()
        
        # Convert object columns to category for efficiency
        for df in [train_metadata, val_metadata, test_metadata]:
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')
        
        train_metadata.to_csv(os.path.join(output_dir, "train_metadata.csv"), index=False)
        val_metadata.to_csv(os.path.join(output_dir, "val_metadata.csv"), index=False)
        test_metadata.to_csv(os.path.join(output_dir, "test_metadata.csv"), index=False)
    
    # Save encoder if provided
    if encoder is not None:
        with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
    
    # Create split info
    split_info = {
        "train_size": float(X_train.shape[0]/(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])),
        "val_size": float(X_val.shape[0]/(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])),
        "test_size": float(X_test.shape[0]/(X_train.shape[0] + X_val.shape[0] + X_test.shape[0])),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_features": int(X_train.shape[1]),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add country distribution if encoder is provided
    if encoder is not None:
        train_countries = encoder.inverse_transform(y_train)
        val_countries = encoder.inverse_transform(y_val)
        test_countries = encoder.inverse_transform(y_test)
        
        country_distribution = {
            "train": dict(Counter(train_countries)),
            "val": dict(Counter(val_countries)),
            "test": dict(Counter(test_countries))
        }
        
        split_info["country_distribution"] = country_distribution
        split_info["countries"] = list(encoder.classes_)
    
    # Add data fingerprint for versioning
    if metadata is not None:
        data_fingerprint = hashlib.md5(pd.util.hash_pandas_object(metadata).values).hexdigest()
        split_info["data_fingerprint"] = data_fingerprint
    
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

def load_splits(input_dir):
    """
    Load previously created splits from disk.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the splits
        
    Returns:
    --------
    dict : Dictionary containing:
        - X_train, X_val, X_test: Feature matrices
        - y_train, y_val, y_test: Labels
        - train_metadata, val_metadata, test_metadata: Metadata DataFrames (if available)
        - encoder: Label encoder (if available)
        - split_info: Split information
    """
    result = {}
    
    # Load feature matrices
    result["X_train"] = load_npz(os.path.join(input_dir, "train_features.npz"))
    result["X_val"] = load_npz(os.path.join(input_dir, "val_features.npz"))
    result["X_test"] = load_npz(os.path.join(input_dir, "test_features.npz"))
    
    # Load labels if available
    if os.path.exists(os.path.join(input_dir, "train_labels.npy")):
        result["y_train"] = np.load(os.path.join(input_dir, "train_labels.npy"))
        result["y_val"] = np.load(os.path.join(input_dir, "val_labels.npy"))
        result["y_test"] = np.load(os.path.join(input_dir, "test_labels.npy"))
    
    # Load metadata if available
    if os.path.exists(os.path.join(input_dir, "train_metadata.csv")):
        result["train_metadata"] = pd.read_csv(os.path.join(input_dir, "train_metadata.csv"))
        result["val_metadata"] = pd.read_csv(os.path.join(input_dir, "val_metadata.csv"))
        result["test_metadata"] = pd.read_csv(os.path.join(input_dir, "test_metadata.csv"))
        
        # Extract labels from metadata if not loaded separately
        if "y_train" not in result and "encoded_country" in result["train_metadata"].columns:
            result["y_train"] = result["train_metadata"]["encoded_country"].values
            result["y_val"] = result["val_metadata"]["encoded_country"].values
            result["y_test"] = result["test_metadata"]["encoded_country"].values
    
    # Load encoder if available
    if os.path.exists(os.path.join(input_dir, "label_encoder.pkl")):
        with open(os.path.join(input_dir, "label_encoder.pkl"), "rb") as f:
            result["encoder"] = pickle.load(f)
    
    # Load split info if available
    if os.path.exists(os.path.join(input_dir, "split_info.json")):
        with open(os.path.join(input_dir, "split_info.json"), "r") as f:
            result["split_info"] = json.load(f)
    
    return result

def main():
    """
    Main function to split and save the preprocessed data.
    Only creates a new split if one with the requested proportions doesn't exist.
    """
    # Load data
    metadata, X, y, sample_ids, encoder = load_data()
    
    # Define split proportions
    test_size = 0.15
    val_size = 0.15
    train_size = 1 - test_size - val_size
    random_state = 42
    
    # Validate split proportions
    if test_size + val_size >= 1.0:
        raise ValueError("The sum of test_size and val_size must be less than 1.0.")
    
    # Create output directory
    split_dir = os.path.join("data", "split")
    try:
        os.makedirs(split_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {split_dir}: {e}")
        return
    
    # Generate data fingerprint for versioning
    data_fingerprint = hashlib.md5(pd.util.hash_pandas_object(metadata).values).hexdigest()
    
    # Check if split already exists with requested proportions
    split_info_path = os.path.join(split_dir, "split_info.json")
    if os.path.exists(split_info_path):
        try:
            with open(split_info_path, "r") as f:
                existing_info = json.load(f)
            
            # Check data fingerprint first to detect data changes
            if existing_info.get("data_fingerprint") == data_fingerprint:
                logging.info(f"Found existing split with same data fingerprint")
                logging.info(f"Using existing split in {split_dir}")
                return
                
            # Otherwise check proportions
            existing_train = existing_info.get("train_size")
            existing_val = existing_info.get("val_size")
            existing_test = existing_info.get("test_size")
            
            if not all(isinstance(x, (int, float)) for x in [existing_train, existing_val, existing_test]):
                raise ValueError("Split proportions must be numeric.")
            
            # Check if existing split has same proportions (with small tolerance for floating point differences)
            if (abs(existing_train - train_size) < 0.01 and 
                abs(existing_val - val_size) < 0.01 and
                abs(existing_test - test_size) < 0.01):
                logging.info(f"Found existing split with {existing_train:.0%}/{existing_val:.0%}/{existing_test:.0%} train/val/test ratio")
                logging.info(f"Data has changed, re-creating the split with same proportions")
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to read or validate split_info.json: {e}")
            logging.info("Proceeding to create a new split.")
        except IOError as e:
            logging.error(f"Failed to open or read split_info.json: {e}")
            return
    
    logging.info(f"Loading data from processed directory...")
    logging.info(f"Found {len(metadata)} samples with {X.shape[1]} features")
    logging.info(f"Countries represented: {len(encoder.classes_)}")
    
    # Check if split is feasible
    try:
        check_split_feasibility(y, test_size=test_size, val_size=val_size, min_samples_per_class=5)
    except ValueError as e:
        logging.error(f"Error: {e}")
        logging.error("Data split is not feasible with the current configuration.")
        return
    
    # Perform the split
    logging.info(f"Splitting data with {train_size:.0%}/{val_size:.0%}/{test_size:.0%} train/val/test ratio...")
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = split_data(
        X, y, sample_ids, test_size=test_size, val_size=val_size, random_state=random_state, stratify=True
    )
    
    # Report split sizes
    logging.info(f"Split sizes:")
    logging.info(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    logging.info(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    logging.info(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    # Save all split data with additional metadata
    logging.info(f"Saving split data to {split_dir}...")
    try:
        save_splits(
            output_dir=split_dir, 
            X_train=X_train, X_val=X_val, X_test=X_test, 
            y_train=y_train, y_val=y_val, y_test=y_test,
            idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
            metadata=metadata, 
            encoder=encoder
        )
    except IOError as e:
        logging.error(f"Failed to save split data: {e}")
        return
    
    logging.info(f"Data splitting complete. Files saved to {split_dir}")
    logging.info(f"You can now proceed to model training using the split data.")

if __name__ == "__main__":
    main()