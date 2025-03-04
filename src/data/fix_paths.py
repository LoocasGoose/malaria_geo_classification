# import os
# import gcsfs.core

# # Save the original _process_path method
# original_process_path = gcsfs.core.GCSFileSystem._process_path

# # Define a patched version that forces forward slashes
# def patched_process_path(self, path, *args, **kwargs):
#     if path and isinstance(path, str):
#         path = path.replace('\\', '/')
#     return original_process_path(self, path, *args, **kwargs)

# # Apply the patch
# gcsfs.core.GCSFileSystem._process_path = patched_process_path

import os
import gcsfs

# Force anonymous access to public buckets
os.environ["GCSFS_ANONYMOUS"] = "true"

# Initialize GCSFS with no credentials
fs = gcsfs.GCSFileSystem(token="anon", access="read_only")

# Force forward slashes for GCS paths
os.environ["FSSPEC_URL_SEPARATOR"] = "/"

# Monkey-patch GCSFileSystem to handle Windows paths
original_open = gcsfs.GCSFileSystem.open

def patched_open(self, path, *args, **kwargs):
    path = path.replace("\\", "/")  # Replace backslashes with forward slashes
    return original_open(self, path, *args, **kwargs)

gcsfs.GCSFileSystem.open = patched_open