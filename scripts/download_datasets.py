#!/usr/bin/env python3
"""
Script to download and setup datasets for pepper_scene_graphs.
Downloads from a Nextcloud share and unzips recursively into the datasets folder.
"""

import os
import zipfile
import urllib.request
from pathlib import Path

# Dataset configuration
DATASET_URL = "https://cloud.vi.cs.uni-bonn.de/index.php/s/e8n2Ne6w6wJeisZ/download"
WORKSPACE_ROOT = Path(__file__).parent.parent.absolute()
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
TEMP_ZIP = DATASETS_DIR / "datasets_download.zip"

def recursive_unzip(zip_path, extract_dir):
    """Unzip a file and recursively unzip any zip files found inside."""
    print(f"Unzipping {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Check for nested zips
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.zip'):
                nested_zip = Path(root) / file
                recursive_unzip(nested_zip, root)
                os.remove(nested_zip)

def main():
    # Ensure datasets directory exists
    DATASETS_DIR.mkdir(exist_ok=True)

    # Simple check for existing data (e.g., check for 'processed' folder)
    if (DATASETS_DIR / "processed").exists():
        print("Dataset 'processed' folder already exists. Skipping download.")
        # Optional: prompt user or add --force flag if needed
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    try:
        urllib.request.urlretrieve(DATASET_URL, TEMP_ZIP)
        print("Download complete.")
        
        recursive_unzip(TEMP_ZIP, DATASETS_DIR)
        
        if TEMP_ZIP.exists():
            os.remove(TEMP_ZIP)
            
        print(f"Dataset successfully setup in {DATASETS_DIR}")
        
    except Exception as e:
        print(f"Error during dataset setup: {e}")
        if TEMP_ZIP.exists():
            os.remove(TEMP_ZIP)

if __name__ == "__main__":
    main()
