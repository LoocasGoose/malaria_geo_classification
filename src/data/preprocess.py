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

'''
def load_sequence_data(fasta_path):
    """
    Load genomic sequences from FASTA file using memory-efficient parsing.
    
    Parameters:
        fasta_path (str): Path to the FASTA file
        
    Returns:
        dict: Dictionary mapping sequence IDs to sequences
    """
    logging.info(f"Loading sequences from {fasta_path}")
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq).upper()
    logging.info(f"Loaded {len(sequences)} sequences")
    return sequences

def filter_sequences(sequences, max_ambiguous_ratio=0.01):
    """
    Filter sequences based on quality criteria.
    
    Parameters:
        sequences (dict): Dictionary mapping sequence IDs to sequences
        max_ambiguous_ratio (float): Maximum allowed ratio of ambiguous bases
        
    Returns:
        dict: Filtered dictionary of sequences
    """
    ambiguous_bases = set("NRYWSMKHBVD")
    filtered_sequences = {}
    
    # More efficient implementation using count method instead of iterating
    for seq_id, sequence in sequences.items():
        # Sum ambiguous base counts more efficiently
        ambiguous_count = sum(sequence.count(base) for base in ambiguous_bases)
        if ambiguous_count / len(sequence) <= max_ambiguous_ratio:
            filtered_sequences[seq_id] = sequence
    
    logging.info(f"Filtered out {len(sequences) - len(filtered_sequences)} sequences")
    return filtered_sequences


def load_metadata(metadata_path):
    """
    Load sample metadata containing geographic information.
    
    Parameters:
        metadata_path (str): Path to the metadata file
        
    Returns:
        pandas.DataFrame: DataFrame containing sample metadata
    """
    logging.info(f"Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path, sep='\t')
    logging.info(f"Loaded metadata for {len(metadata)} samples")
    return metadata

def extract_metadata(metadata):
    metadata = metadata[['Sample', 'Country', 'Admin level 1', 
                        'Country latitude', 'Country longitude', 
                        'Admin level 1 latitude', 
                        'Admin level 1 longitude', 
                        'Year', 'Population', '% callable', 
                        'QC pass', 'Sample type', 'Exclusion reason',
                        'All samples same case']]
    logging.info(f"Extracted relevent columns from original metadata")
    return metadata

    !!!
    K-MER GENERATION IS NO LONGER USED BY CONVOLUTIONAL NEURAL NETWORK. 
    !!!
    
def generate_kmers(sequences, k=6):
    """
    Generate k-mer frequency vectors from DNA sequences.
    
    Parameters:
        sequences (list): List of DNA sequences
        k (int): k-mer length
        
    Returns:
        tuple: (sparse matrix of k-mer counts, vectorizer object)
    """
    # Check if sequences list is empty
    if not sequences:
        logging.error("Cannot generate k-mers: sequence list is empty")
        raise ValueError("Empty sequence list provided to generate_kmers")
    
    # Log some info about the sequences
    logging.info(f"Generating {k}-mers from {len(sequences)} sequences")
    avg_length = sum(len(s) for s in sequences) / len(sequences)
    logging.info(f"Average sequence length: {avg_length:.2f}")
    
    # Dynamically set min_df based on dataset size
    # For small datasets, use absolute count, for larger use percentage
    if len(sequences) < 100:
        min_df_value = 2  # Minimum absolute count for very small datasets
    elif len(sequences) < 1000:
        min_df_value = max(2, int(0.01 * len(sequences)))  # 1% for small/medium datasets
    else:
        min_df_value = max(5, int(0.001 * len(sequences)))  # 0.1% for large datasets
        
    logging.info(f"Using min_df={min_df_value} (k-mer must appear in at least {min_df_value} sequences)")
    
    try:
        # Set max_features to prevent memory issues with very large datasets
        max_features = 100000 if len(sequences) > 5000 else None
        
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(k, k),
            lowercase=False,
            min_df=min_df_value,
            max_features=max_features
        )

        # Process in batches if the dataset is very large
        if len(sequences) > 10000 and avg_length > 1000:
            return _generate_kmers_batched(sequences, vectorizer)
        else:
            X_counts = vectorizer.fit_transform(sequences)
            logging.info(f"Generated {len(vectorizer.get_feature_names_out())} unique {k}-mers")
            return X_counts, vectorizer
            
    except ValueError as e:
        logging.error(f"Error generating k-mers: {str(e)}")
        logging.info("Trying with min_df=1 (include all k-mers)")
        # Fallback to min_df=1 if the first attempt fails
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(k, k),
            lowercase=False,
            min_df=1  # Include all k-mers
        )
        X_counts = vectorizer.fit_transform(sequences)
        logging.info(f"Generated {len(vectorizer.get_feature_names_out())} unique {k}-mers with min_df=1")
        return X_counts, vectorizer

def _generate_kmers_batched(sequences, vectorizer, batch_size=1000):
    """Process k-mers in batches to save memory for large datasets"""
    logging.info(f"Using batched k-mer generation with batch size {batch_size}")
    
    # First fit on all sequences to get the vocabulary
    vectorizer.fit(sequences)
    feature_count = len(vectorizer.get_feature_names_out())
    logging.info(f"Vocabulary size: {feature_count} k-mers")
    
    # Then transform in batches
    result_parts = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:min(i+batch_size, len(sequences))]
        batch_counts = vectorizer.transform(batch)
        result_parts.append(batch_counts)
        if (i // batch_size) % 10 == 0:
            logging.info(f"Processed {i+len(batch)}/{len(sequences)} sequences")
    
    # Combine results
    X_counts = sparse.vstack(result_parts)
    return X_counts, vectorizer
'''

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

'''
def cache_reference_genome():
    """Cache reference genome locally for faster access."""
    pf7 = malariagen_data.Pf7()
    
    # Define contig_info inside the function
    contig_info = pf7.genome_sequence().attrs["contigs"]
    
    cache_dir = os.path.join("data", "reference")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save reference genome by chromosome
    for chrom, length in zip(contig_info["id"], contig_info["length"]):
        chrom_seq = pf7.genome_sequence(region=chrom).compute()
        np.save(os.path.join(cache_dir, f"{chrom}.npy"), chrom_seq)
    
    logging.info(f"Reference genome cached in {cache_dir}")
'''

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