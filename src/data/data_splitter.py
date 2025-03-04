'''
data_splitter.py
~~~~~~~~~~
'''
import os
import pandas as pd
import pickle
import json
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
import numpy as np
from collections import Counter
from datetime import datetime

metadata = pd.read_csv("data/processed/filtered_metadata.csv")
y = metadata["encoded_country"].values
sample_ids = metadata["Sample"].values
X = load_npz("data/processed/variant_features.npz")
with open("data/processed/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

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
    # First split: Test set
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        features, labels, sample_ids, 
        test_size=test_size, 
        stratify=stratify,
        random_state=random_state
    )

    # Second split: Validation from remaining
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val,
        test_size=val_size/(1-test_size),  # val_size = test_size * (1-test_size) / test_size
        stratify=stratify,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test

def stratified_split(features, labels, test_size, val_size, random_state):
    """
    Perform a stratified split preserving the distribution of geographic origins.
    This will use scikit-learn's train_test_split with the stratify parameter.
    """
    # First split: Test set
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        features, labels, sample_ids, 
        test_size=0.15, 
        stratify=labels,
        random_state=42
    )

    # Second split: Validation from remaining
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val,
        test_size=0.1765,  # 0.15/(1-0.15) = ~17.65% of original
        stratify=labels,
        random_state=42
    )


def check_split_feasibility(labels, test_size, val_size, min_samples_per_class=1):
    """
    Check if the requested split is possible given the geographic distribution.
    Some regions might have too few samples to be represented in all splits.
    """


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
        train_metadata = metadata.loc[metadata["Sample"].isin(idx_train)].copy()
        val_metadata = metadata.loc[metadata["Sample"].isin(idx_val)].copy()
        test_metadata = metadata.loc[metadata["Sample"].isin(idx_test)].copy()
        
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
    """
    # Create output directory
    split_dir = os.path.join("data", "split")
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Loading data from processed directory...")
    print(f"Found {len(metadata)} samples with {X.shape[1]} features")
    print(f"Countries represented: {len(encoder.classes_)}")
    
    # Check if split is feasible
    check_split_feasibility(y, test_size=0.15, val_size=0.15, min_samples_per_class=5)
    
    # Perform the split
    print(f"Splitting data with 70/15/15 train/val/test ratio...")
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = split_data(
        X, y, sample_ids, test_size=0.15, val_size=0.15, random_state=42, stratify=True
    )
    
    # Report split sizes
    print(f"Split sizes:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    # Save all split data with additional metadata
    print(f"Saving split data to {split_dir}...")
    save_splits(
        output_dir=split_dir, 
        X_train=X_train, X_val=X_val, X_test=X_test, 
        y_train=y_train, y_val=y_val, y_test=y_test,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        metadata=metadata, 
        encoder=encoder
    )
    
    print(f"Data splitting complete. Files saved to {split_dir}")
    print(f"You can now proceed to model training using the split data.")

if __name__ == "__main__":
    main()