# import os
# import fsspec  # Import first to configure
# import malariagen_data

# # Force forward slashes using environment variables
# os.environ["FSSPEC_URL_SEPARATOR"] = "/"

# # Try direct configuration (different methods since fsspec.config.set doesn't exist)
# try:
#     # Method 1: Dictionary-style assignment
#     fsspec.config["url_separator"] = "/"
# except:
#     try:
#         # Method 2: Direct attribute assignment
#         fsspec.config.url_separator = "/"
#     except:
#         print("Warning: Could not set fsspec config, relying on environment variables")

# print("Attempting to access Pf7 data...")

# # Try multiple initialization methods
# methods = [
#     {
#         "name": "GCS with explicit path",
#         "params": {"url": "gs://pf7_release/", "use_gcs": True}
#     },
#     {
#         "name": "HTTP fallback",
#         "params": {"url": "https://github.com/malariagen", "use_gcs": False}
#     },
#     {
#         "name": "Default initialization",
#         "params": {}
#     }
# ]

# success = False
# for method in methods:
#     print(f"\nTrying method: {method['name']}")
#     try:
#         pf7 = malariagen_data.Pf7(**method['params'])
#         metadata = pf7.sample_metadata()
#         print(f"SUCCESS! Found {len(metadata)} samples")
#         print(f"Sample columns: {list(metadata.columns)[:5]}...")
#         success = True
#         print(f"Use these parameters in preprocess.py: {method['params']}")
#         break
#     except Exception as e:
#         print(f"Failed with error: {str(e)}")

# if not success:
#     print("\nAll methods failed. Consider manual data download instead.")
#     print("Visit: https://www.malariagen.net/resource/26")

# import os
# import fsspec
# from fix_paths import *  # Apply path fixes first
# import malariagen_data

# # Initialize with corrected paths
# pf7 = malariagen_data.Pf7(
#     url="gcs://pf7_release/metadata",  # Explicit forward slashes
#     use_gcs=True
# )

# # Test access
# metadata = pf7.sample_metadata()
# print(f"Success! Found {len(metadata)} samples")

from fix_paths import fs  # Apply fix first
import malariagen_data

pf7 = malariagen_data.Pf7(
    url="gcs://pf7_release/metadata",
    storage_options={"token": "anon"}  # Use anonymous access
)

metadata = pf7.sample_metadata()
print(f"Success! Found {len(metadata)} samples")
