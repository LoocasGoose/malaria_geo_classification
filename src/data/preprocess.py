"""
Genomic Data Preprocessing Module for Malaria Geographic Origin Classification.

This module handles the preprocessing of genomic data from the MalariaGEN Pf7 dataset, 
preparing it for machine learning model training. Key preprocessing steps include:

1. Loading and filtering metadata based on quality thresholds
2. Extracting variant information for selected chromosomes
3. Processing variants into feature vectors using TF-IDF transformation
4. Encoding geographic labels for classification
5. Saving processed data for downstream model training

Design choices:
- We use TF-IDF vectorization to capture the relative importance of genetic variants
- We apply batched processing to handle large genomic datasets efficiently
- We filter samples based on metadata quality to ensure reliable training data
- We focus on specific chromosomes that have been identified as informative for geographic classification

References:
- MalariaGEN Pf7: https://www.malariagen.net/parasite/pf7
- TF-IDF for genomic data: Sundararajan et al. (2018), DOI: 10.1093/bioinformatics/bty237
"""

import os
import logging
from Bio import SeqIO
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import malariagen_data
from scipy import sparse
import xarray as xr
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_tfidf_batched(binary_matrix, batch_size=5000):
    """
    Apply TF-IDF transformation to a large binary feature matrix using batched processing.
    
    This function handles large sparse matrices by processing them in manageable batches,
    which prevents memory issues when working with genomic-scale data. TF-IDF transformation
    is used to highlight variants that are distinctive to particular geographic regions
    while downweighting commonly occurring variants.
    
    Args:
        binary_matrix (scipy.sparse.csr_matrix): Binary matrix of variant presence/absence
            with shape [n_samples, n_features]
        batch_size (int, optional): Number of samples to process in each batch.
            Default is 5000, which balances memory usage and processing efficiency.
    
    Returns:
        scipy.sparse.csr_matrix: TF-IDF transformed feature matrix with same shape as input
        
    Design choice:
        Batched processing is critical for genomic data where feature matrices can exceed
        available RAM. The batch size can be adjusted based on available memory.
    """
    n_samples = binary_matrix.shape[0]
    transformer = TfidfTransformer(norm='l2', smooth_idf=True)
    
    # Fit on entire dataset - log memory usage
    logging.info(f"Fitting TF-IDF transformer on {n_samples} samples with {binary_matrix.shape[1]} features")
    mem_usage_mb = binary_matrix.data.nbytes / (1024 * 1024)
    logging.info(f"Input matrix memory usage: {mem_usage_mb:.2f} MB")
    
    # Dynamically adjust batch size based on matrix size
    if mem_usage_mb > 1000:  # If matrix is over 1GB
        batch_size = min(batch_size, max(100, int(5000 * (1000 / mem_usage_mb))))
        logging.info(f"Adjusted batch size to {batch_size} based on memory usage")
        
    transformer.fit(binary_matrix)
    
    # Transform in batches
    result_parts = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = binary_matrix[start:end]
        transformed_batch = transformer.transform(batch)
        result_parts.append(transformed_batch)
        logging.info(f"Transformed batch {start//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
    
    # Combine results
    return sparse.vstack(result_parts)

def encode_labels(labels, encoder = None):
    """
    Encode categorical geographic labels to numeric values for model training.
    
    Converts string country names into numeric indices for classification tasks.
    Can either create a new encoder or use a provided one for consistent encoding
    between training and test data.
    
    Args:
        labels (array-like): Geographic origin labels (typically country names)
        encoder (sklearn.preprocessing.LabelEncoder, optional): Existing encoder for consistent
            encoding between datasets. If None, a new encoder is created. Default is None.
    
    Returns:
        tuple: (encoded_labels, encoder)
            - encoded_labels (numpy.ndarray): Numeric labels
            - encoder (sklearn.preprocessing.LabelEncoder): The encoder used
    
    Design choice:
        We return both the encoded labels and the encoder itself to ensure
        consistent encoding between training, validation, and test sets.
    """
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labels)
        
    encoded_labels = encoder.transform(labels)
    return encoded_labels, encoder

def main():
    """
    Run the complete preprocessing pipeline for malaria genomic data.
    
    This function orchestrates the entire preprocessing workflow:
    1. Configure processing parameters
    2. Initialize data access to the MalariaGEN Pf7 dataset
    3. Load and filter metadata based on quality thresholds
    4. Encode geographic labels for classification
    5. Process genetic variants to create feature vectors
    6. Save all processed data for downstream model training
    
    The function uses helper methods to modularize each processing step.
    
    Returns:
        None: Results are saved to disk in the configured output directory
        
    Design choices:
        - Selected chromosomes are limited to reduce dimensionality while preserving signal
        - Quality threshold ensures only high-quality samples are used
        - Minimum samples per class ensures balanced representation
    """
    # Create configuration with defaults that can be overridden
    config = {
        "metadata_quality_threshold": 0.5,  # Minimum % callable for quality filter
        "min_samples_per_class": 50,        # Minimum samples needed per country
        "selected_chroms": ["Pf3D7_01_v3", "Pf3D7_04_v3", "Pf3D7_07_v3", "Pf3D7_13_v3"],
        "output_dir": os.path.join("data", "processed"),
        "cache_dir": os.path.join("data", "reference"),
        "use_gcs": True                     # Use Google Cloud Storage
    }
    
    # Create output directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["cache_dir"], exist_ok=True)
    
    logging.info("Initializing Pf7 data access...")
    os.environ['FSSPEC_URL_SEPARATOR'] = '/'

    # Initialize with explicit GCS protocol
    pf7 = malariagen_data.Pf7(use_gcs=config["use_gcs"])
    
    try:
        # Load and filter metadata in a separate function to keep main() cleaner
        filtered_metadata = _load_and_filter_metadata(pf7, config)
        
        # Encode geographic labels
        country_labels = filtered_metadata["country"].values
        encoded_labels, label_encoder = encode_labels(country_labels)
        
        # Add encoded labels back to DataFrame
        filtered_metadata['encoded_country'] = encoded_labels
        
        # Process variants
        tfidf_features = _process_variants(pf7, filtered_metadata, config)
        
        # Save processed data
        _save_outputs(filtered_metadata, tfidf_features, label_encoder, config)
        
        # Save selected chromosomes for downstream tasks
        with open(os.path.join(config["output_dir"], "selected_chromosomes.json"), "w") as f:
            json.dump(config["selected_chroms"], f)
            
        logging.info("Preprocessing completed successfully.")
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        logging.error("Please check your internet connection and retry")
        import traceback
        logging.error(traceback.format_exc())
        return

def _load_and_filter_metadata(pf7, config):
    """
    Load and filter sample metadata based on quality criteria and class balance.
    
    This function:
    1. Loads the raw sample metadata from MalariaGEN Pf7
    2. Applies quality filters to remove low-quality samples
    3. Ensures geographic balance by filtering classes with too few samples
    4. Logs detailed information about the filtering process
    
    Args:
        pf7 (malariagen_data.Pf7): Initialized Pf7 data access object
        config (dict): Configuration parameters including quality thresholds
    
    Returns:
        pandas.DataFrame: Filtered metadata for high-quality samples
        
    Design choice:
        Quality filtering is essential for genomic data, as low-quality samples can
        introduce noise that obscures true geographic signals. The min_samples_per_class
        threshold ensures we have sufficient examples from each region for effective learning.
    """
    try:
        # This will initialize data access and might download files if not already present
        sample_metadata = pf7.sample_metadata()
        logging.info(f"Successfully loaded metadata for {len(sample_metadata)} samples")
        
        # Filter for quality
        quality_metadata = sample_metadata[
            (sample_metadata["QC pass"] == True) &
            (sample_metadata["% callable"] >= config["metadata_quality_threshold"])
        ].copy()
        logging.info(f"After quality filtering, {len(quality_metadata)} samples remain")
        
        # Get country distribution
        country_counts = quality_metadata['country'].value_counts()
        logging.info(f"Sample distribution by country:\n{country_counts.head(10)}")
        
        # Check if we have enough samples per country for classification
        valid_countries = country_counts[country_counts >= config["min_samples_per_class"]].index
        logging.info(f"Found {len(valid_countries)} countries with at least {config['min_samples_per_class']} samples")
        
        # Filter metadata to include only countries with sufficient samples
        filtered_metadata = quality_metadata[quality_metadata['country'].isin(valid_countries)].copy()
        logging.info(f"Working with {len(filtered_metadata)} samples from {len(valid_countries)} countries")
        
        return filtered_metadata
    except Exception as e:
        logging.error(f"Error loading metadata: {str(e)}")
        raise

def _process_variants(pf7, filtered_metadata, config):
    """
    Extract and process genetic variants from selected chromosomes into feature vectors.
    
    This function:
    1. Loads variant data for each selected chromosome
    2. Converts variants to a binary presence/absence matrix
    3. Combines features from all chromosomes
    4. Applies TF-IDF transformation to weight features
    
    Args:
        pf7 (malariagen_data.Pf7): Initialized Pf7 data access object
        filtered_metadata (pandas.DataFrame): Filtered sample metadata
        config (dict): Configuration parameters including selected chromosomes
    
    Returns:
        scipy.sparse.csr_matrix: TF-IDF transformed feature matrix for all samples
        
    Design choice:
        We process variants chromosome by chromosome to manage memory usage,
        and apply TF-IDF to highlight geographic-specific variants while
        downweighting common variants. Focusing on selected chromosomes
        reduces dimensionality while preserving discriminative power.
    """
    try:
        logging.info("Loading variant calls...")
        variant_data = pf7.variant_calls(extended=False)
        variant_data = variant_data.sel(samples=filtered_metadata.index)
        
        # Get chromosome names to filter variants
        contig_info = pf7.genome_sequence().attrs["contigs"]
        chrom_names = contig_info["id"]
        chrom_indices = {name: idx for idx, name in enumerate(chrom_names)}
        
        # Process selected chromosomes
        selected_chroms = config["selected_chroms"]
        all_variant_features = []
        
        for chrom in selected_chroms:
            logging.info(f"Processing {chrom}...")
            chrom_idx = chrom_indices[chrom]
            # Filter variants by chromosome
            mask = variant_data["variant_contig"] == chrom_idx
            chrom_variants = variant_data.sel(variant=mask)
            
            # Extract genotypes and convert to binary presence/absence
            genotypes = chrom_variants["call_genotype"].data.compute()
            variant_presence = (genotypes > 0).any(axis=2).astype(np.int8)
            variant_presence = sparse.csr_matrix(variant_presence.T)  # Transpose to samples x variants
            
            all_variant_features.append(variant_presence)
        
        # Combine all variant features
        combined_features = sparse.hstack(all_variant_features, format='csr')
        logging.info(f"Total variants: {combined_features.shape[1]}")
        
        # Apply TF-IDF transformation
        logging.info("Applying TF-IDF transformation...")
        tfidf_features = apply_tfidf_batched(combined_features)
        
        return tfidf_features
    except Exception as e:
        logging.error(f"Error processing variants: {str(e)}")
        raise

def _save_outputs(filtered_metadata, tfidf_features, label_encoder, config):
    """
    Save all preprocessed data to disk for downstream model training.
    
    This function saves:
    1. Filtered metadata as CSV
    2. TF-IDF feature matrix as sparse NPZ file
    3. Label encoder as pickled object
    
    Args:
        filtered_metadata (pandas.DataFrame): Filtered sample metadata
        tfidf_features (scipy.sparse.csr_matrix): TF-IDF transformed feature matrix
        label_encoder (sklearn.preprocessing.LabelEncoder): Encoder for geographic labels
        config (dict): Configuration parameters including output directory
        
    Returns:
        None: Files are saved to disk
        
    Design choice:
        We save the sparse feature matrix in the NPZ format to preserve memory efficiency.
        The label encoder is saved to ensure consistent encoding between training and inference.
    """
    try:
        # Save metadata
        filtered_metadata.to_csv(os.path.join(config["output_dir"], "filtered_metadata.csv"), index=False)
        
        # Save TF-IDF features
        sparse.save_npz(os.path.join(config["output_dir"], "variant_features.npz"), tfidf_features)
        
        # Save label encoder
        with open(os.path.join(config["output_dir"], "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)
            
        logging.info(f"All preprocessed data saved to {config['output_dir']}")
    except Exception as e:
        logging.error(f"Error saving outputs: {str(e)}")
        raise

if __name__ == "__main__":
    main()