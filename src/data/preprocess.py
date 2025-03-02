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
pf7 = malariagen_data.Pf7()

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
    # Define paths - adjust these to match your actual project structure
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    
    metadata_path = os.path.join(raw_dir, "Pf7_samples.txt")
    fasta_path = os.path.join(raw_dir, "Pfalciparum.genome.fasta")
    
    # Check if files exist
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata file not found: {metadata_path}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Please make sure the file exists or adjust the path")
        return
    
    if not os.path.exists(fasta_path):
        logging.error(f"FASTA file not found: {fasta_path}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Please make sure the file exists or adjust the path")
        return
    
    # Load metadata and sequences
    metadata = load_metadata(metadata_path)
    metadata = extract_metadata(metadata)
    
    # After loading metadata and sequences, add these debug statements:
    logging.info(f"Loaded metadata with {len(metadata)} rows")
    logging.info(f"Metadata columns: {list(metadata.columns)}")
    logging.info(f"First few 'Sample' values: {metadata['Sample'].head().tolist()}")
    
    # Filter metadata for quality
    quality_metadata = metadata[metadata["QC pass"] == True].copy()
    
    # Load and filter sequences
    sequences = load_sequence_data(fasta_path)
    logging.info(f"Loaded {len(sequences)} sequences from FASTA file")
    logging.info(f"First few sequence IDs: {list(sequences.keys())[:5]}")
    
    # Check sequence ID format
    if sequences and metadata.shape[0] > 0:
        sample_seq_id = list(sequences.keys())[0]
        sample_metadata_id = metadata['Sample'].iloc[0]
        logging.info(f"Sample sequence ID format: '{sample_seq_id}'")
        logging.info(f"Sample metadata ID format: '{sample_metadata_id}'")
    
    # After filtering sequences
    filtered_sequences = filter_sequences(sequences, max_ambiguous_ratio=0.01)
    logging.info(f"After filtering, {len(filtered_sequences)} sequences remain")
    
    # After filtering metadata for quality
    quality_metadata = metadata[metadata["QC pass"] == True].copy()
    logging.info(f"After quality filtering, {len(quality_metadata)} metadata rows remain")
    
    # Check for ID matching issues before attempting to map
    if len(quality_metadata) > 0 and len(filtered_sequences) > 0:
        # Check if any sample IDs match sequence IDs
        sample_ids = set(quality_metadata['Sample'].tolist())
        sequence_ids = set(filtered_sequences.keys())
        common_ids = sample_ids.intersection(sequence_ids)
        
        logging.info(f"Number of sample IDs that match sequence IDs: {len(common_ids)}")
        
        if len(common_ids) == 0:
            logging.error("No matching IDs found between metadata and sequences!")
            logging.info("This might be due to formatting differences in IDs.")
            
            # Try a simple transformation to see if it helps
            logging.info("Attempting to find matches with simple transformations...")
            
            # Option 1: Try lowercasing both
            lower_sample_ids = {s.lower() for s in sample_ids}
            lower_seq_ids = {s.lower() for s in sequence_ids}
            common_lower = lower_sample_ids.intersection(lower_seq_ids)
            logging.info(f"Matches after lowercasing: {len(common_lower)}")
            
            # Option 2: Try removing any prefixes/suffixes
            # This assumes IDs might have format differences like "sample_12345" vs "12345"
            # Extract numeric parts only
            import re
            numeric_sample_ids = {re.sub(r'[^0-9]', '', s) for s in sample_ids if re.search(r'\d', s)}
            numeric_seq_ids = {re.sub(r'[^0-9]', '', s) for s in sequence_ids if re.search(r'\d', s)}
            common_numeric = numeric_sample_ids.intersection(numeric_seq_ids)
            logging.info(f"Matches when comparing only numeric parts: {len(common_numeric)}")
            
            # Print a few examples to help diagnose the issue
            logging.info(f"Sample ID examples: {list(sample_ids)[:5]}")
            logging.info(f"Sequence ID examples: {list(sequence_ids)[:5]}")
            
            # If transformations found matches, offer a suggestion
            if len(common_lower) > 0:
                logging.info("Consider using lowercase transformation for matching.")
            if len(common_numeric) > 0:
                logging.info("Consider extracting numeric parts for matching.")
    
    # Continue with existing code...
    quality_metadata['Sequence'] = quality_metadata['Sample'].map(filtered_sequences)
    matched_data = quality_metadata.dropna(subset=['Sequence'])
    
    logging.info(f"Successfully matched {len(matched_data)} samples with sequences")
    
    if len(matched_data) == 0:
        logging.error("No samples could be matched with sequences - cannot proceed.")
        return
    
    # Generate k-mer features from the sequences
    sequence_list = matched_data['Sequence'].tolist()
    kmer_counts, vectorizer = generate_kmers(sequence_list, k=6)
    kmer_tfidf, tfidf_transformer = apply_tfidf(kmer_counts)
    
    # Encode geographic labels for classification
    country_labels = matched_data["Country"].tolist()
    encoded_labels, label_encoder = encode_labels(country_labels)
    
    # Add the encoded labels back to the DataFrame
    matched_data['EncodedCountry'] = encoded_labels
    
    # Save processed data
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save metadata with sequences
    metadata_output = os.path.join(processed_dir, "metadata_with_sequences.csv")
    matched_data.to_csv(metadata_output, index=False)
    
    # Save feature matrices
    import scipy.sparse as sp
    feature_output = os.path.join(processed_dir, "kmer_features.npz")
    sp.save_npz(feature_output, kmer_tfidf)
    
    # Save vectorizer and encoder for future use
    import pickle
    with open(os.path.join(processed_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(processed_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    logging.info(f"Preprocessing complete. Data saved to {processed_dir}")

if __name__ == "__main__":
    main()