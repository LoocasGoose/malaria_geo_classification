"""
Common utility functions for the Malaria Geographic Classification project.

This module provides helper functions used across different project components:
- Configuration management
- Logging setup
- Path handling
- Performance utilities
- Common data transformations
"""

import os
import logging
import yaml
import time
import torch
import numpy as np
from pathlib import Path
from functools import wraps
from typing import Dict, Any, Callable, Union, Optional, List, Tuple

# Configuration Management
def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise

# Logging Setup
def setup_logging(log_dir: str = "logs", 
                  log_level: int = logging.INFO,
                  filename: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        filename: Optional specific log filename
        
    Returns:
        None
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    
    if filename:
        log_path = os.path.join(log_dir, filename)
        handlers.append(logging.FileHandler(log_path))
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(log_dir, f"malaria_{timestamp}.log")
        handlers.append(logging.FileHandler(log_path))
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logging.info(f"Logging configured. Log file: {log_path}")

# Performance Utilities
def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

def get_device(cuda_if_available: bool = True) -> torch.device:
    """
    Get appropriate device (CPU/GPU) for PyTorch.
    
    Args:
        cuda_if_available: Whether to use CUDA if available
        
    Returns:
        torch.device to use for computation
    """
    if cuda_if_available and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict with memory statistics in MB
    """
    memory_stats = {}
    
    # CPU memory via psutil if available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_stats['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        memory_stats['cpu_memory_mb'] = None
    
    # GPU memory via PyTorch
    if torch.cuda.is_available():
        memory_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
    return memory_stats

# Path Handling
def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to ensure exists
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Assuming utils.py is in the src folder
    return Path(__file__).parent.parent

# DNA Sequence Utilities
def reverse_complement(seq: str) -> str:
    """
    Get the reverse complement of a DNA sequence.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        Reverse complemented sequence
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
                 'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def one_hot_encode_dna(seq: str) -> np.ndarray:
    """
    Convert DNA sequence to one-hot encoding.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        One-hot encoded array with shape (len(seq), 5)
        Channels: A, C, G, T, N/other
    """
    # Define mapping
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Initialize array
    one_hot = np.zeros((len(seq), 5), dtype=np.float32)
    
    # Fill array
    for i, base in enumerate(seq):
        if base in base_dict:
            one_hot[i, base_dict[base]] = 1.0
        else:
            # Unknown base (N or other)
            one_hot[i, 4] = 1.0
            
    return one_hot

# Model Utilities
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Metric Utilities
def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights based on label distribution.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of weights for each class
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Weight is inversely proportional to class frequency
    weights = total_samples / (class_counts * len(class_counts))
    
    return torch.tensor(weights, dtype=torch.float)

# Progress Tracking
class ProgressLogger:
    """
    Helper class to log training progress.
    """
    
    def __init__(self, total_epochs: int, total_batches: int, log_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            total_epochs: Total number of epochs
            total_batches: Total number of batches per epoch
            log_interval: How often to log (in batches)
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.log_interval = log_interval
        self.start_time = time.time()
        self.epoch_start_time = self.start_time
        
    def log_batch(self, epoch: int, batch: int, loss: float, 
                 additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log progress for a single batch.
        
        Args:
            epoch: Current epoch
            batch: Current batch
            loss: Current loss value
            additional_metrics: Optional additional metrics to log
        """
        if batch % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            progress = (epoch * self.total_batches + batch) / (self.total_epochs * self.total_batches)
            remaining = elapsed / progress - elapsed if progress > 0 else 0
            
            log_str = (f"Epoch: {epoch}/{self.total_epochs} | "
                     f"Batch: {batch}/{self.total_batches} | "
                     f"Loss: {loss:.6f}")
            
            if additional_metrics:
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in additional_metrics.items()])
                log_str += f" | {metrics_str}"
                
            log_str += f" | Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s"
            
            logging.info(log_str)
            
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                 metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log progress for a completed epoch.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss (optional)
            metrics: Optional additional metrics to log
        """
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_start_time = time.time()
        
        log_str = (f"Epoch {epoch}/{self.total_epochs} completed in {epoch_time:.2f}s | "
                 f"Train Loss: {train_loss:.6f}")
        
        if val_loss is not None:
            log_str += f" | Val Loss: {val_loss:.6f}"
            
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_str += f" | {metrics_str}"
            
        logging.info(log_str)
