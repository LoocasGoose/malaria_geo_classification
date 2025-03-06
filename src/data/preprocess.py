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

def apply_tfidf_batched(binary_matrix, batch_size=5000):
    """Apply TF-IDF transformation in batches to save memory"""
    transformer = TfidfTransformer(norm='l2', smooth_idf=True)
    
    # Fit on entire dataset
    transformer.fit(binary_matrix)
    
    # Transform in batches
    n_samples = binary_matrix.shape[0]
    result_parts = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = binary_matrix[start:end]
        transformed_batch = transformer.transform(batch)
        result_parts.append(transformed_batch)
    
    # Combine results
    return sparse.vstack(result_parts)

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
        quality_metadata = sample_metadata[sample_metadata["QC pass"] == True].copy()
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
        
        # Encode geographic labels
        country_labels = filtered_metadata["country"].values
        encoded_labels, label_encoder = encode_labels(country_labels)
        
        # Add encoded labels back to DataFrame
        filtered_metadata['encoded_country'] = encoded_labels
        
        # Load variant data once
        logging.info("Loading variant calls...")
        variant_data = pf7.variant_calls(extended=False)
        variant_data = variant_data.sel(samples=filtered_metadata.index)
        
        # Get chromosome names to filter variants
        contig_info = pf7.genome_sequence().attrs["contigs"]
        chrom_names = contig_info["id"]
        chrom_indices = {name: idx for idx, name in enumerate(chrom_names)}
        
        # Process selected chromosomes
        selected_chroms = ["Pf3D7_01_v3", "Pf3D7_04_v3", "Pf3D7_07_v3", "Pf3D7_13_v3"]
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
        
        # Save processed data
        processed_dir = os.path.join("data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save metadata
        filtered_metadata.to_csv(os.path.join(processed_dir, "filtered_metadata.csv"), index=False)
        
        # Save TF-IDF features
        sparse.save_npz(os.path.join(processed_dir, "variant_features.npz"), tfidf_features)
        
        # Save label encoder
        with open(os.path.join(processed_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)
        
        logging.info("Preprocessing completed successfully.")
        
    except Exception as e:
        logging.error(f"Error accessing Pf7 data: {str(e)}")
        logging.error("Please check your internet connection and retry")
        import traceback
        logging.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()