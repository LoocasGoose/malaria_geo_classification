import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from scipy.sparse import load_npz
import pickle
import logging
import time
from functools import lru_cache
import dask
from dask.distributed import Client
import hashlib
import json

"""
Genomic Sequence Dataset Module for Malaria Geographic Origin Classification.

This module provides a PyTorch Dataset implementation for dynamically reconstructing
genomic sequences from variant data. Key features include:

1. On-the-fly sequence reconstruction from reference genome and variants
2. Efficient caching mechanisms for both memory and disk
3. Sliding window approach for sequence segmentation
4. One-hot encoding of DNA sequences for neural network input
5. Support for sampling and batch processing

Design choices:
- Dynamic reconstruction reduces storage requirements compared to storing full sequences
- Caching improves performance while managing memory usage
- Sliding windows with configurable stride enable flexible sequence coverage
- One-hot encoding captures the categorical nature of DNA bases
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GenomicSequenceDataset(Dataset):
    """
    PyTorch Dataset for dynamically reconstructing genomic sequences from variants.
    
    This dataset handles the complex task of reconstructing DNA sequences from a reference
    genome and variant data, providing a memory-efficient way to work with genomic data.
    It implements sliding windows to segment long chromosomes into manageable pieces
    and provides one-hot encoded tensors suitable for CNN input.
    
    The dataset supports caching at multiple levels to balance performance and memory usage:
    1. In-memory caching of frequently accessed reference sequences
    2. Optional disk caching for reconstructed sequences
    3. Efficient variant lookup through indexing
    
    It also provides functionality for sampling regions to facilitate development and testing.
    """
    
    def __init__(self, split_dir, split_type='train', window_size=1000, stride=500, 
                 chromosomes=None, cache_size=128, sample_limit=None):
        """
        Initialize dataset with configurable parameters.
        
        Args:
            split_dir (str): Directory containing split data from data_splitter.py
            split_type (str): 'train', 'val', or 'test' split to use
            window_size (int): Size of sequence windows to generate (in base pairs)
            stride (int): Step size between consecutive windows (in base pairs)
            chromosomes (list): List of specific chromosomes to use (e.g., ["Pf3D7_01_v3"])
                               If None, uses all chromosomes from preprocessing
            cache_size (int): Number of reference windows to cache in memory
            sample_limit (int): Limit dataset to this many samples (for testing)
            
        Design choices:
            - Window size of 1000bp balances context size with computational efficiency
            - 50% overlap (stride=500) ensures features at window boundaries are captured
            - Chromosome selection allows focusing on specific regions of interest
            - Memory caching improves performance for repeated access patterns
        """
        self.split_dir = split_dir
        self.split_type = split_type
        self.window_size = window_size
        self.stride = stride
        self.cache_size = cache_size
        
        start_time = time.time()
        logging.info(f"Initializing {split_type} dataset...")
        
        # Load metadata and labels
        self.metadata = pd.read_csv(os.path.join(split_dir, f"{split_type}_metadata.csv"))
        if sample_limit:
            self.metadata = self.metadata.head(sample_limit)
            
        self.sample_ids = self.metadata["Sample"].values
        
        # Try loading compressed labels first, then fall back to .npy
        try:
            if os.path.exists(os.path.join(split_dir, f"{split_type}_labels.npz")):
                self.labels = np.load(os.path.join(split_dir, f"{split_type}_labels.npz"))["labels"]
            else:
                self.labels = np.load(os.path.join(split_dir, f"{split_type}_labels.npy"))
        except Exception as e:
            logging.error(f"Error loading labels: {e}")
            raise
            
        logging.info(f"Loaded {len(self.sample_ids)} samples with labels")
        
        # Load encoder
        with open(os.path.join(split_dir, "label_encoder.pkl"), "rb") as f:
            self.encoder = pickle.load(f)
        
        # Initialize Pf7 API
        import malariagen_data
        self.pf7 = malariagen_data.Pf7()
        
        # Get chromosome info - filter if specified
        if chromosomes is None:
            # Try to load selected chromosomes from preprocessing
            chrom_path = os.path.join(split_dir, "..", "processed", "selected_chromosomes.json")
            if os.path.exists(chrom_path):
                with open(chrom_path, "r") as f:
                    chromosomes = json.load(f)
        self.chromosomes = self._get_chromosome_info(chromosomes)
        logging.info(f"Using {len(self.chromosomes)} chromosomes")
        
        # Set up caching
        self.cache_dir = os.path.join(split_dir, "..", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create reference sequence cache with LRU policy
        self._ref_sequence_cache = {}
        
        # Create sample windows efficiently
        self.sample_windows = self._create_sample_windows_memory_efficient()
        logging.info(f"Created {len(self.sample_windows)} windows across {len(self.chromosomes)} chromosomes")
        
        # Report initialization time
        elapsed = time.time() - start_time
        logging.info(f"Dataset initialization completed in {elapsed:.2f} seconds")
    
    def _get_chromosome_info(self, selected_chroms=None):
        """
        Get information about chromosomes to process.
        
        This function retrieves chromosome information from the reference genome,
        optionally filtering to include only specified chromosomes. It's a critical
        step in dataset initialization that determines which genomic regions will
        be included in the dataset.
        
        Args:
            selected_chroms (list): List of chromosome names to include.
                                   If None, all chromosomes are included.
                                   
        Returns:
            list: List of dictionaries with chromosome information:
                 - name: Chromosome name
                 - length: Chromosome length in base pairs
                 - windows: Number of windows that will be created
                 
        Design choice:
            Filtering to specific chromosomes allows focusing on regions known to be
            informative for geographic classification, reducing computational requirements
            while maintaining classification performance.
        """
        # Get chromosome info from reference genome
        genome_info = self.pf7.genome_features()
        chromosomes = []
        
        for chrom in genome_info:
            # Skip if not in selected chromosomes
            if selected_chroms is not None and chrom["name"] not in selected_chroms:
                continue
                
            # Calculate number of windows
            n_windows = max(1, (chrom["length"] - self.window_size) // self.stride + 1)
            
            chromosomes.append({
                "name": chrom["name"],
                "length": chrom["length"],
                "windows": n_windows
            })
        
        if not chromosomes:
            if selected_chroms:
                raise ValueError(f"None of the selected chromosomes {selected_chroms} were found")
            else:
                raise ValueError("No chromosomes found in reference genome")
                
        return chromosomes
    
    def _create_sample_windows_memory_efficient(self):
        """
        Create a list of all sample windows across all chromosomes.
        
        This function generates the complete list of (sample_idx, chrom, start, end) tuples
        that define all windows to be processed. It's designed to be memory-efficient
        by generating windows on-the-fly rather than storing all possible sequences.
        
        Returns:
            list: List of tuples (sample_idx, chrom, start, end) for all windows
            
        Design choice:
            This approach allows handling very large datasets with many samples and
            chromosomes without excessive memory usage. The window generation is
            deterministic, ensuring reproducibility across runs.
        """
        windows = []
        
        # For each sample
        for sample_idx in range(len(self.sample_ids)):
            # For each chromosome
            for chrom_info in self.chromosomes:
                chrom = chrom_info["name"]
                chrom_length = chrom_info["length"]
                
                # Create windows with specified stride
                for window_idx in range(chrom_info["windows"]):
                    start = window_idx * self.stride
                    end = min(start + self.window_size, chrom_length)
                    
                    # Skip windows that are too small
                    if end - start < self.window_size * 0.9:  # Allow some flexibility
                        continue
                        
                    windows.append((sample_idx, chrom, start, end))
        
        return windows
    
    def __len__(self):
        """Return the total number of windows in the dataset."""
        return len(self.sample_windows)
    
    def __getitem__(self, idx):
        """
        Get a sequence window with its label.
        
        This is the core method that PyTorch's DataLoader calls to retrieve items.
        It dynamically reconstructs a genomic sequence for the requested window,
        applies one-hot encoding, and returns the sequence along with its label
        and metadata.
        
        Args:
            idx (int): Index of the window to retrieve
            
        Returns:
            dict: Contains:
                - 'sequence': One-hot encoded tensor of shape [window_size, 5]
                - 'label': Class label (geographic origin)
                - 'region': String identifier of the genomic region
                
        Design choice:
            Returning a dictionary allows for flexible expansion of returned data
            without breaking the API. The one-hot encoding is performed here rather
            than in the model to simplify the model architecture.
        """
        # Get window coordinates
        sample_idx, chrom, start, end = self.sample_windows[idx]
        sample_id = self.sample_ids[sample_idx]
        label = self.labels[sample_idx]
        
        # Get reference sequence for this window
        ref_seq = self._get_ref_sequence(chrom, start, end)
        
        # Get variants for this sample in this window
        variants = self._get_variants(chrom, start, end, sample_id)
        
        # Apply variants to the reference sequence
        seq = self._apply_variants_optimized(ref_seq, variants, start)
        
        # One-hot encode the sequence
        seq_tensor = self._one_hot_encode(seq)
        
        return {'sequence': seq_tensor, 'label': label, 'region': f"{chrom}:{start}-{end}"}
    
    def _get_ref_sequence_uncached(self, chrom, start, end):
        """
        Get reference sequence for a window (uncached version).
        
        This function retrieves the reference sequence directly from the Pf7 API
        without caching. It's used as a fallback when a sequence is not in the cache.
        
        Args:
            chrom (str): Chromosome name
            start (int): Start position (0-based)
            end (int): End position (exclusive)
            
        Returns:
            str: DNA sequence string
            
        Note:
            The Pf7 API uses 1-based coordinates, so we adjust the start position.
        """
        return self.pf7.genome_sequence(region=f"{chrom}:{start+1}-{end}").compute()
    
    @lru_cache(maxsize=128)
    def _get_ref_sequence(self, chrom, start, end):
        """
        Get reference sequence for a window with caching.
        
        This function retrieves the reference sequence with LRU caching to improve
        performance for frequently accessed regions. It's decorated with lru_cache
        to automatically handle the caching logic.
        
        Args:
            chrom (str): Chromosome name
            start (int): Start position (0-based)
            end (int): End position (exclusive)
            
        Returns:
            str: DNA sequence string
            
        Design choice:
            LRU caching significantly improves performance when the same reference
            regions are accessed repeatedly, which is common in training with multiple
            epochs. The cache size is configurable to balance memory usage and performance.
        """
        return self._get_ref_sequence_uncached(chrom, start, end)
    
    def _get_variants(self, chrom, start, end, sample_id):
        """
        Get variants for a specific sample in a genomic window.
        
        This function retrieves variant data for a specific sample in a specific
        genomic region. It filters the variants to include only those within the
        specified window and belonging to the specified sample.
        
        Args:
            chrom (str): Chromosome name
            start (int): Start position (0-based)
            end (int): End position (exclusive)
            sample_id (str): Sample identifier
            
        Returns:
            list: List of variant dictionaries with:
                - POS: Position (1-based)
                - REF: Reference allele
                - ALT: Alternate allele
                - sample_idx: Index of the sample in the variant call
                
        Design choice:
            Retrieving variants for specific windows rather than whole chromosomes
            significantly reduces memory usage and improves performance, especially
            for large datasets with many variants.
        """
        try:
            # Get variants in this region
            variants = self.pf7.variant_calls(
                region=f"{chrom}:{start+1}-{end}",
                sample_ids=[sample_id],
                field="GT"
            ).compute()
            
            # Extract positions, reference and alternate alleles
            positions = variants.POS.values
            ref_alleles = variants.REF.values
            alt_alleles = variants.ALT.values
            
            # Get genotypes for this sample (should be only one sample)
            genotypes = variants.GT.values[:, 0, :]  # First dimension is variant, second is sample, third is ploidy
            
            # Create list of variants that this sample has
            sample_variants = []
            
            for i in range(len(positions)):
                # Check if sample has the variant (any non-zero genotype)
                if np.any(genotypes[i] > 0):
                    # Get the specific alternate allele(s) this sample has
                    alt_indices = genotypes[i][genotypes[i] > 0] - 1  # -1 because 1 = first alt, 2 = second alt, etc.
                    
                    # Add each alternate allele as a separate variant
                    for alt_idx in alt_indices:
                        sample_variants.append({
                            "POS": positions[i],  # 1-based position
                            "REF": ref_alleles[i],
                            "ALT": alt_alleles[i][alt_idx],
                            "sample_idx": 0  # We only have one sample here
                        })
            
            return sample_variants
        
        except Exception as e:
            logging.error(f"Error getting variants for {sample_id} at {chrom}:{start}-{end}: {e}")
            return []  # Return empty list on error
    
    def _apply_variants_optimized(self, ref_seq, variants, window_start):
        """
        Apply variants to a reference sequence efficiently.
        
        This function applies a list of variants to a reference sequence to reconstruct
        the sample-specific sequence. It handles substitutions, insertions, and deletions
        in a memory-efficient way by processing the sequence in a single pass.
        
        Args:
            ref_seq (str): Reference sequence
            variants (list): List of variant dictionaries
            window_start (int): Start position of the window (0-based)
            
        Returns:
            str: Reconstructed sequence with variants applied
            
        Design choice:
            This optimized implementation handles complex cases like overlapping
            variants and indels efficiently. Processing variants in position order
            ensures correctness while maintaining performance.
        """
        if not variants:
            return ref_seq
        
        # Sort variants by position
        variants = sorted(variants, key=lambda v: v["POS"])
        
        # Convert to 0-based positions relative to window
        for v in variants:
            v["rel_pos"] = v["POS"] - 1 - window_start
        
        # Filter out variants outside the window
        variants = [v for v in variants if 0 <= v["rel_pos"] < len(ref_seq)]
        
        if not variants:
            return ref_seq
        
        # Apply variants one by one, adjusting positions for indels
        result = list(ref_seq)  # Convert to list for efficient character replacement
        pos_shift = 0  # Track position shift due to indels
        
        for v in variants:
            # Adjust position based on previous indels
            adj_pos = v["rel_pos"] + pos_shift
            
            # Skip if position is now outside the sequence
            if adj_pos < 0 or adj_pos >= len(result):
                continue
            
            # Get reference and alternate alleles
            ref_allele = v["REF"]
            alt_allele = v["ALT"]
            
            # Handle different variant types
            ref_len = len(ref_allele)
            alt_len = len(alt_allele)
            
            # Check if reference matches (basic validation)
            if ''.join(result[adj_pos:adj_pos + ref_len]) != ref_allele:
                # This can happen with overlapping variants or data issues
                # For robustness, we'll skip this variant
                continue
            
            # Apply the variant
            if alt_len <= ref_len:
                # Substitution or deletion: replace characters directly
                for i in range(alt_len):
                    result[adj_pos + i] = alt_allele[i]
                
                # For deletions, remove extra characters
                if alt_len < ref_len:
                    del result[adj_pos + alt_len:adj_pos + ref_len]
                    pos_shift -= (ref_len - alt_len)
            else:
                # Insertion: replace ref part and insert additional characters
                for i in range(ref_len):
                    result[adj_pos + i] = alt_allele[i]
                
                # Insert additional characters
                for i in range(ref_len, alt_len):
                    result.insert(adj_pos + i, alt_allele[i])
                
                pos_shift += (alt_len - ref_len)
        
        return ''.join(result)
    
    def _one_hot_encode(self, seq):
        """
        Convert DNA sequence to one-hot encoded tensor.
        
        This function converts a DNA sequence string to a one-hot encoded tensor
        suitable for input to neural networks. It handles the standard DNA bases
        (A, C, G, T) and maps any other characters to a fifth channel (N).
        
        Args:
            seq (str): DNA sequence string
            
        Returns:
            torch.Tensor: One-hot encoded tensor of shape [seq_length, 5]
            
        Design choice:
            One-hot encoding is the standard representation for categorical data
            in neural networks. Including a fifth channel for N (unknown) bases
            ensures the model can handle real-world genomic data with ambiguous bases.
        """
        # Define mapping from nucleotides to indices
        nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Initialize tensor with zeros
        one_hot = torch.zeros((len(seq), 5), dtype=torch.float32)
        
        # Fill in one-hot encoding
        for i, nuc in enumerate(seq.upper()):
            if nuc in nuc_to_idx:
                one_hot[i, nuc_to_idx[nuc]] = 1.0
            else:
                # For any non-standard nucleotide (N, etc.), use the fifth channel
                one_hot[i, 4] = 1.0
        
        return one_hot
    
    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        """
        Convenience method to create a DataLoader for this dataset.
        
        This method creates a PyTorch DataLoader with appropriate settings for
        efficient batch processing of genomic sequences.
        
        Args:
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for this dataset
            
        Design choice:
            Providing this convenience method simplifies the API for users and
            ensures consistent DataLoader configuration across different uses.
            The default parameters are optimized for typical use cases.
        """
        from torch.utils.data import DataLoader
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True  # Faster data transfer to GPU
        )

    def _get_cache_path(self, chrom, start, end, sample_id):
        """
        Generate a unique cache path for a sequence window.
        
        This function creates a unique file path for caching a specific sequence
        window for a specific sample. It uses a hash of the window coordinates
        and sample ID to ensure uniqueness and reasonable path lengths.
        
        Args:
            chrom (str): Chromosome name
            start (int): Start position
            end (int): End position
            sample_id (str): Sample identifier
            
        Returns:
            str: Path to the cache file
            
        Design choice:
            Using a hash-based filename ensures uniqueness while keeping path
            lengths manageable, which is important for file systems with path
            length limitations.
        """
        key = f"{sample_id}_{chrom}_{start}_{end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.npy")

    def _get_or_create_sequence(self, chrom, start, end, sample_id, use_disk_cache=False):
        """
        Get sequence from cache or create it.
        
        This function implements a two-level caching strategy:
        1. Check disk cache (if enabled)
        2. Generate sequence by applying variants to reference
        
        It's used to avoid redundant sequence reconstruction for the same
        genomic regions and samples.
        
        Args:
            chrom (str): Chromosome name
            start (int): Start position
            end (int): End position
            sample_id (str): Sample identifier
            use_disk_cache (bool): Whether to use disk caching
            
        Returns:
            numpy.ndarray: One-hot encoded sequence
            
        Design choice:
            The two-level caching strategy balances performance and memory usage.
            Disk caching is optional and can be enabled for very large datasets
            where memory caching would be insufficient.
        """
        if use_disk_cache:
            cache_path = self._get_cache_path(chrom, start, end, sample_id)
            if os.path.exists(cache_path):
                return np.load(cache_path)
        
        # Get reference and apply variants
        ref_seq = self._get_ref_sequence(chrom, start, end)
        variants = self._get_variants(chrom, start, end, sample_id)
        seq = self._apply_variants_optimized(ref_seq, variants, start)
        
        # Cache to disk if requested
        if use_disk_cache:
            cache_path = self._get_cache_path(chrom, start, end, sample_id)
            np.save(cache_path, seq)
        
        return seq

    def sample_regions(self, n_regions=100):
        """
        Randomly sample a subset of regions for faster development and testing.
        
        This function reduces the dataset size by randomly sampling a specified
        number of regions. It's useful for development, debugging, and initial
        model testing where using the full dataset would be too time-consuming.
        
        Args:
            n_regions (int): Number of regions to sample
            
        Returns:
            None: Modifies the dataset in-place
            
        Design choice:
            Random sampling preserves the distribution of the data while reducing
            size, which is important for getting representative results even with
            a smaller dataset. This is more effective than simply truncating the
            dataset, which could introduce bias.
        """
        if n_regions >= len(self.sample_windows):
            logging.info(f"Requested {n_regions} regions but dataset only has {len(self.sample_windows)}. Using all regions.")
            return
        
        # Sample random indices without replacement
        indices = np.random.choice(len(self.sample_windows), size=n_regions, replace=False)
        
        # Update sample windows
        self.sample_windows = [self.sample_windows[i] for i in indices]
        
        logging.info(f"Sampled {len(self.sample_windows)} regions for faster development")
        
        # Report class distribution after sampling
        sample_indices = [sw[0] for sw in self.sample_windows]
        unique_samples = len(set(sample_indices))
        
        logging.info(f"Sampled regions cover {unique_samples} unique samples")
        
        if hasattr(self, 'labels') and len(self.labels) > 0:
            # Get labels for sampled regions
            region_labels = [self.labels[sw[0]] for sw in self.sample_windows]
            
            # Count occurrences of each label
            label_counts = {}
            for label in region_labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            
            # Log label distribution
            logging.info(f"Label distribution after sampling:")
            for label, count in sorted(label_counts.items()):
                if hasattr(self, 'encoder') and hasattr(self.encoder, 'classes_'):
                    label_name = self.encoder.classes_[label]
                    logging.info(f"  {label_name}: {count} regions ({count/len(self.sample_windows):.1%})")
                else:
                    logging.info(f"  Class {label}: {count} regions ({count/len(self.sample_windows):.1%})")
