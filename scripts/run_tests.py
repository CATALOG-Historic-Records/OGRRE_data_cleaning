import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the test suite."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    
    # Run pytest with coverage
    result = subprocess.run(
        ["pytest", "--cov=ogrre_data_cleaning", "--cov-report=term-missing"],
        cwd=project_root
    )
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests()) 