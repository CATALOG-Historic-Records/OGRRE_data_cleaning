import os
from pathlib import Path
import hashlib
import requests
from typing import Optional
from tqdm import tqdm

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_URLS = {
    ("holesize", "0"): {
        "url": "https://github.com/CATALOG-Historic-Records/OGRRE_data_cleaning/releases/download/v0.1.0/checkpoint-holesize-0.pt",
        "md5": "b63a44470cb0b7b2f74a0f48bc71729c"
    }
}

class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""
    pass

class CheckpointDownloadError(CheckpointError):
    """Raised when checkpoint download fails."""
    pass

class CheckpointVerificationError(CheckpointError):
    """Raised when checkpoint verification fails."""
    pass

def get_checkpoint_path(model_name: str, version: str) -> Path:
    """Get or download model checkpoint."""
    checkpoint_path = CHECKPOINTS_DIR / f"checkpoint-{model_name}-{version}.pt"
    
    if not checkpoint_path.exists():
        download_checkpoint(model_name, version, checkpoint_path)
    
    return checkpoint_path

def download_checkpoint(model_name: str, version: str, target_path: Path):
    """Download checkpoint file if it doesn't exist."""
    if (model_name, version) not in CHECKPOINT_URLS:
        raise CheckpointError(
            f"No checkpoint found for {model_name} version {version}. "
            f"Available models: {list(CHECKPOINT_URLS.keys())}"
        )
    
    info = CHECKPOINT_URLS[(model_name, version)]
    
    try:
        print(f"Downloading checkpoint for {model_name} v{version}...")
        response = requests.get(info["url"], stream=True)
        response.raise_for_status()
        
        # Download with progress bar
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        
        with open(target_path, "wb") as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
        
        # Verify checksum
        if info["md5"]:
            print("Verifying download...")
            file_hash = hashlib.md5(open(target_path, "rb").read()).hexdigest()
            if file_hash != info["md5"]:
                target_path.unlink()  # Delete corrupted file
                raise CheckpointVerificationError(
                    f"Downloaded file has incorrect checksum\n"
                    f"Expected: {info['md5']}\n"
                    f"Got: {file_hash}"
                )
            print("Verification successful!")
        
    except requests.RequestException as e:
        raise CheckpointDownloadError(
            f"Failed to download checkpoint: {str(e)}"
        ) from e
    except Exception as e:
        if target_path.exists():
            target_path.unlink()  # Clean up partial download
        raise 

def clear_checkpoint_cache():
    """Remove all downloaded checkpoints."""
    for checkpoint in CHECKPOINTS_DIR.glob("*.pt"):
        checkpoint.unlink()
    print("Checkpoint cache cleared.")

def get_cache_info():
    """Get information about cached checkpoints."""
    cached_files = list(CHECKPOINTS_DIR.glob("*.pt"))
    total_size = sum(f.stat().st_size for f in cached_files)
    
    print(f"Checkpoint cache directory: {CHECKPOINTS_DIR}")
    print(f"Number of cached checkpoints: {len(cached_files)}")
    print(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")
    
    for f in cached_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  - {f.name}: {size_mb:.1f} MB") 