import hashlib
from pathlib import Path

def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

if __name__ == "__main__":
    checkpoints_dir = Path("./checkpoints")
    for checkpoint in checkpoints_dir.glob("*.pt"):
        md5 = calculate_md5(checkpoint)
        print(f"{checkpoint.name}: {md5}") 