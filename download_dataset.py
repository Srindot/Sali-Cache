import os

# Set the download path BEFORE importing kagglehub
os.environ['KAGGLEHUB_CACHE'] = "/ssd_scratch/vishnu/"

import kagglehub

# Now, this will download to /ssd_scratch/vishnu/datasets/valerytamrazov/msrvttqa
path = kagglehub.dataset_download("valerytamrazov/msrvttqa")

# This will print the *actual* path where the files are
print("Dataset downloaded to:", path)