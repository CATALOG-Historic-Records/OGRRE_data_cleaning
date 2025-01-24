import subprocess
import sys
import venv
from pathlib import Path

def test_install():
    """Test package installation in a fresh virtual environment."""
    # Create a temporary venv
    venv_dir = Path("temp_venv")
    venv.create(venv_dir, with_pip=True)
    
    # Get python path
    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"
    
    try:
        # Install package
        subprocess.run(
            [str(python_path), "-m", "pip", "install", "."],
            check=True
        )
        
        # Try importing and basic usage
        test_code = """
import ogrre_data_cleaning as odc
result = odc.string_to_float('123.45')
assert result == 123.45
print('Installation test passed!')
        """
        
        subprocess.run(
            [str(python_path), "-c", test_code],
            check=True
        )
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(venv_dir)

if __name__ == "__main__":
    test_install() 