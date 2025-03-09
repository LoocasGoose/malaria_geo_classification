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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GenomicSequenceDataset(Dataset):
    """PyTorch Dataset for dynamically reconstructing genomic sequences from variants."""
    
    def __init__(self, split_dir, split_type='train', window_size=1000, stride=500, 
                 chromosomes=None, cache_size=128, sample_limit=None):
        """
        Initialize dataset with configurable parameters.
        
        Args:
            split_dir: Directory containing split data
            split_type: 'train', 'val', or 'test'
            window_size: Size of sequence windows to generate
            stride: Step size between windows
            chromosomes: List of specific chromosomes to use (e.g., ["Pf3D7_01_v3"])
                         If None, uses all chromosomes (can be very large)
            cache_size: Number of reference windows to cache
            sample_limit: Limit dataset to this many samples (for testing)
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
        
        # Set up caching for reference sequences
        self._get_ref_sequence = lru_cache(maxsize=cache_size)(self._get_ref_sequence_uncached)
        
        # Create mapping of sample indices to windows
        self.sample_windows = self._create_sample_windows_memory_efficient()
        logging.info(f"Created {len(self.sample_windows)} windows")
        
        init_time = time.time() - start_time
        logging.info(f"Dataset initialization completed in {init_time:.2f} seconds")
        
        # Print memory usage warning if dataset is large
        est_memory_mb = len(self.sample_windows) * self.window_size * 5 / (1024 * 1024)
        if est_memory_mb > 1000:  # More than 1GB
            logging.warning(f"This dataset could potentially use {est_memory_mb:.0f} MB of memory "
                           f"if all windows were loaded at once. Ensure you have sufficient RAM.")
        
    def _get_chromosome_info(self, selected_chroms=None):
        """Get information about chromosomes, optionally filtering to a subset."""
        contig_info = self.pf7.genome_sequence().attrs["contigs"]
        all_chroms = [(name, length) for name, length in zip(contig_info["id"], contig_info["length"])]
        
        if selected_chroms:
            # Filter to only the requested chromosomes
            return [(name, length) for name, length in all_chroms if name in selected_chroms]
        return all_chroms
    
    def _create_sample_windows_memory_efficient(self):
        """Create windows with memory optimization."""
        total_windows = 0
        for _, length in self.chromosomes:
            total_windows += (length - self.window_size) // self.stride + 1
        total_windows *= len(self.sample_ids)
        
        # Pre-allocate with numpy for memory efficiency
        window_data = np.zeros((total_windows, 4), dtype=object)
        
        idx = 0
        for i, sample_id in enumerate(self.sample_ids):
            for chrom, length in self.chromosomes:
                for start in range(0, length - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    window_data[idx] = [i, chrom, start, end]
                    idx += 1
        
        return window_data[:idx]
    
    def __len__(self):
        """Return the number of windows across all samples."""
        return len(self.sample_windows)
    
    def __getitem__(self, idx):
        """Get a sequence window with its label.
        
        Returns:
            dict: Contains 'sequence' (one-hot encoded tensor) and 'label' (class)
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
        """Get reference sequence for a window (uncached version)."""
        return self.pf7.genome_sequence(region=f"{chrom}:{start+1}-{end}").compute()
    
    def _get_variants(self, chrom, start, end, sample_id):
        """Get variants for a specific sample in a window."""
        try:
            # Query for variants in this region
            variants = self.pf7.variant_calls(
                region=f"{chrom}:{start+1}-{end}", 
                samples=[sample_id]
            ).compute()
            return variants
        except Exception as e:
            logging.error(f"Error getting variants for {chrom}:{start}-{end}, sample {sample_id}: {e}")
            # Return an empty dataset or handle error appropriately
            return None
    
    def _apply_variants_optimized(self, ref_seq, variants, window_start):
        """Apply variants to reference sequence using vectorized operations."""
        if variants is None or len(variants.variant_position) == 0:
            return ref_seq
        
        seq = ref_seq.copy()
        
        # Get positions and genotypes
        positions = variants.variant_position.values
        alt_alleles = variants.variant_allele.values
        genotypes = variants.call_genotype.values[0]
        
        # Convert to relative positions within window
        rel_positions = positions - (window_start + 1)
        
        # Create mask for positions within window
        valid_mask = (rel_positions >= 0) & (rel_positions < len(seq))
        
        # Create mask for non-reference genotypes
        alt_mask = np.array([any(gt > 0 for gt in gts) for gts in genotypes])
        
        # Combine masks
        combined_mask = valid_mask & alt_mask
        
        # Apply variants using vectorized operations
        for i in np.where(combined_mask)[0]:
            rel_pos = rel_positions[i]
            seq[rel_pos] = alt_alleles[i, 1]  # First alternate allele
        
        return seq
    
    def _one_hot_encode(self, seq):
        """One-hot encode DNA sequence.
        
        Maps A, C, G, T, and N to one-hot vectors.
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        seq_numeric = np.array([mapping.get(base.upper(), 4) for base in seq])
        
        # One-hot encode (4 bases + N)
        one_hot = np.zeros((len(seq), 5), dtype=np.float32)
        for i, base in enumerate(seq_numeric):
            one_hot[i, base] = 1.0
            
        return torch.tensor(one_hot, dtype=torch.float32)
    
    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        """Convenience method to create a DataLoader for this dataset."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True  # Faster data transfer to GPU
        )

    def _get_cache_path(self, chrom, start, end, sample_id):
        """Generate a unique cache path for a sequence window."""
        key = f"{sample_id}_{chrom}_{start}_{end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.npy")

    def _get_or_create_sequence(self, chrom, start, end, sample_id, use_disk_cache=False):
        """Get sequence from cache or create it."""
        if use_disk_cache:
            cache_path = self._get_cache_path(chrom, start, end, sample_id)
            if os.path.exists(cache_path):
                return np.load(cache_path)
        
        # Get reference and apply variants
        ref_seq = self._get_ref_sequence(chrom, start, end)
        variants = self._get_variants(chrom, start, end, sample_id)
        seq = self._apply_variants(ref_seq, variants, start)
        
        # Cache to disk if requested
        if use_disk_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, seq)
        
        return seq

    def sample_regions(self, n_regions=100):
        """Sample a subset of regions for faster testing."""
        if n_regions >= len(self.sample_windows):
            return
        
        # Ensure at least one window per sample
        sample_indices = {}
        for i, (sample_idx, _, _, _) in enumerate(self.sample_windows):
            if sample_idx not in sample_indices:
                sample_indices[sample_idx] = []
            sample_indices[sample_idx].append(i)
        
        # Sample regions evenly across samples
        sampled_indices = []
        samples_per_region = max(1, n_regions // len(sample_indices))
        
        for indices in sample_indices.values():
            if len(indices) <= samples_per_region:
                sampled_indices.extend(indices)
            else:
                sampled_indices.extend(np.random.choice(indices, samples_per_region, replace=False))
        
        # Trim if needed
        if len(sampled_indices) > n_regions:
            sampled_indices = np.random.choice(sampled_indices, n_regions, replace=False)
        
        # Update sample windows
        self.sample_windows = [self.sample_windows[i] for i in sampled_indices]
        logging.info(f"Sampled {len(self.sample_windows)} windows for testing")
