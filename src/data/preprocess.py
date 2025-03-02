'''
preprocess.py
~~~~~~~~~~
'''
import os
import logging
from Bio import SeqIO
import pandas as pd
import numpy as np

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

def reconstruct_sequences():


def generate_kmers():


def encode_labels():


def main():

    
if __name__ == "__main__":
    main()