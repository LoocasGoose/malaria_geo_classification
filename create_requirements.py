#!/usr/bin/env python

"""
Script to generate requirements.txt file for the Malaria Geographic Classification project.
"""

requirements = [
    # Core dependencies
    "numpy>=1.22.0",
    "pandas>=1.4.0",
    "scipy>=1.8.0",
    "scikit-learn>=1.0.2",
    "torch>=1.12.0",
    "biopython>=1.79",
    
    # Data handling
    "xarray>=2022.3.0",
    "fsspec>=2022.3.0",
    "gcsfs>=2022.3.0",
    "malariagen_data>=0.1.0",
    "zarr>=2.11.0",
    "dask>=2022.3.0",
    
    # Visualization
    "matplotlib>=3.5.1",
    "seaborn>=0.11.2",
    "plotly>=5.6.0",
    
    # Utilities
    "tqdm>=4.64.0",
    "joblib>=1.1.0",
    "optuna>=2.10.0",
    "tensorboard>=2.8.0",
]

with open("requirements.txt", "w") as f:
    for req in requirements:
        f.write(f"{req}\n")

print("requirements.txt file created successfully!") 