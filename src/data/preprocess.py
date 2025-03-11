'''
preprocess.py
~~~~~~~~~~
'''
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
    """Apply TF-IDF transformation in batches to save memory"""
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
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labels)
        
    encoded_labels = encoder.transform(labels)
    return encoded_labels, encoder

def main():
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
    """Load and filter metadata based on quality criteria"""
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
    """Process variant data for selected chromosomes"""
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
    """Save all preprocessed data with error handling"""
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