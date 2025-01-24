# OGRRE Data Cleaning

A Python package for cleaning OGRRE data.

## Installation
To install from source

1. Clone the repository
2. Navigate to the repository directory
3. Run the following command:

```bash
pip install .
```

## Usage

To use the package, import the functions you need from the `clean` module.

```python
from ogrre_data_cleaning import string_to_float, string_to_int, llm_clean
```

Note: Model checkpoints will be automatically downloaded when first using the `llm_clean` function.