'''
preprocess.py
~~~~~~~~~~
'''
import os
import logging
from Bio import SeqIO
import pandas as pd
import numpy as np
from scikit-learn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scikit-learn.preprocessing import LabelEncoder

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
    logging.info(f"Extracted relevent columnes from original metadata")
    return metadata

def generate_kmers(sequences, k = 6):
    vectorizer = CountVectorizer(
    analyzer='char',
    ngram_range=(k, k),
    lowercase=False,
    min_df=0.01 
    )

    X_counts = vectorizer.fit_transform(sequences)

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
    # metadata processing
    metadata = load_metadata("data\raw\Pf7_samples.txt")
    metadata = extract_metadata(metadata)
    final_metadata = metadata[metadata["QC pass"] == True]

    # sequence processing
    sequences = load_sequence_data("data\raw\Pfalciparum.genome.fasta")
    clean_sequences = [seq for seq in sequences if "N" not in seq]
    
    #matching metadata to sequence
    metadata['Sequence'] = metadata['Sample_ID'].map(sequences)
    metadata = metadata.dropna(subset=['Sequence'])

    #generate k-mer and apply TF-IDF weighting to k-mer counts
    kmer_counts, vectorizer = generate_kmers(clean_sequences, k=6)
    kmer_tfodf, tfidf_transformer = apply_tfidf(kmer_counts)

    labels = encode_labels(final_metadata["Country"])

    output_path = "data/processed/metadata_with_sequences.csv"
    metadata.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()