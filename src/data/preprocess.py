'''
preprocess.py
~~~~~~~~~~
'''
import os
import logging
from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import malariagen_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    for seq_id, sequence in sequences.items():
        ambiguous_count = sum(1 for base in sequence if base in ambiguous_bases)
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
    logging.info(f"Average sequence length: {sum(len(s) for s in sequences) / len(sequences):.2f}")
    
    # Set min_df based on number of sequences - use a more appropriate threshold
    # For small datasets, use absolute count instead of percentage
    min_df_value = 2 if len(sequences) < 100 else max(2, int(0.001 * len(sequences)))
    logging.info(f"Using min_df={min_df_value} (k-mer must appear in at least {min_df_value} sequences)")
    
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(k, k),
        lowercase=False,
        min_df=min_df_value  # Use the calculated value instead of fixed 0.01
    )

    try:
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


def apply_tfidf(X_counts):
    tfidf_transformer = TfidfTransformer(
        norm='l2',  # Normalize vectors
        smooth_idf=True  # Prevent divide-by-zero
    )
    
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    
    return X_tfidf, tfidf_transformer

def encode_labels(labels, encoder = None):
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labels)
        
    encoded_labels = encoder.transform(labels)
    return encoded_labels, encoder

def main():
    logging.info("Initializing Pf7 data access...")
    os.environ['FSSPEC_URL_SEPARATOR'] = '/'

    # Initialize with explicit GCS protocol
    pf7 = malariagen_data.Pf7(use_gcs=True)    
    # Use the malariagen_data package to access data
    try:
        # This will initialize data access and might download files if not already present
        sample_metadata = pf7.sample_metadata()
        logging.info(f"Successfully loaded metadata for {len(sample_metadata)} samples")
        logging.info(f"Metadata columns: {list(sample_metadata.columns)}")
        
        # Filter for quality
        quality_metadata = sample_metadata[sample_metadata["qc_pass"] == True].copy()
        logging.info(f"After quality filtering, {len(quality_metadata)} samples remain")
        
        # Get country distribution
        country_counts = quality_metadata['country'].value_counts()
        logging.info(f"Sample distribution by country:\n{country_counts.head(10)}")
        
        # Check if we have enough samples per country for classification
        min_samples_per_class = 50  # Minimum samples needed for a reliable classification model
        valid_countries = country_counts[country_counts >= min_samples_per_class].index
        logging.info(f"Found {len(valid_countries)} countries with at least {min_samples_per_class} samples")
        
        # Filter metadata to include only countries with sufficient samples
        filtered_metadata = quality_metadata[quality_metadata['country'].isin(valid_countries)].copy()
        logging.info(f"Working with {len(filtered_metadata)} samples from {len(valid_countries)} countries")
        
        # Process genomic data
        # Option 1: Access variant data (SNPs)
        # This gives you access to genetic variants rather than full sequences
        # Variant data might be more appropriate for classification tasks
        
        # Example: Get variants for chromosome 1
        logging.info("Accessing variant data for chromosome 1...")
        variants = pf7.snp_genotypes(region="Pf3D7_01_v3", sample_selection=filtered_metadata.index)
        logging.info(f"Loaded {variants.shape[1]} variants for chromosome 1")
        
        # Convert variant data to features
        # Here we'll create a simple binary feature matrix (presence/absence of variants)
        # For each sample (row) and variant position (column)
        logging.info("Converting variant data to feature matrix...")
        variant_features = variants.to_n_alt().compute()  # Convert to number of alternate alleles (0, 1, 2)
        variant_binary = (variant_features > 0).astype(int)  # Convert to binary (variant present/absent)
        
        # Use scikit-learn to apply TF-IDF weighting
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf_transformer = TfidfTransformer(norm='l2', smooth_idf=True)
        variant_tfidf = tfidf_transformer.fit_transform(variant_binary)
        
        # Encode geographic labels
        country_labels = filtered_metadata["country"].values
        encoded_labels, label_encoder = encode_labels(country_labels)
        
        # Add encoded labels back to DataFrame
        filtered_metadata['encoded_country'] = encoded_labels
        
        # Save processed data
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save metadata
        metadata_output = os.path.join(processed_dir, "filtered_metadata.csv")
        filtered_metadata.to_csv(metadata_output, index=False)
        
        # Save feature matrix
        import scipy.sparse as sp
        feature_output = os.path.join(processed_dir, "variant_features.npz")
        sp.save_npz(feature_output, variant_tfidf)
        
        # Save encoder
        import pickle
        with open(os.path.join(processed_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)
        
        logging.info(f"Preprocessing complete. Data saved to {processed_dir}")
        
    except Exception as e:
        logging.error(f"Error accessing Pf7 data: {str(e)}")
        logging.error("Please check your internet connection and retry")
        import traceback
        logging.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()