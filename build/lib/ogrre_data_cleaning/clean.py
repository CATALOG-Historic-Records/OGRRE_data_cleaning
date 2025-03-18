import re
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

from ogrre_data_cleaning.models.encoder import Encoder, Classifier
from ogrre_data_cleaning.models.dataloaders import HoleSize
from ogrre_data_cleaning.models.checkpoints import get_checkpoint_path

def string_to_date(s: str):
    """
    Converts a string to a date after removing non-date characters.
    """
    # If input is already a datetime object, return it
    if isinstance(s, datetime):
        return s
        
    if not isinstance(s, str):
        return None
    
    # Use regex to keep only valid date characters
    cleaned_string = re.sub(r"[^\d-]+", "", s)
    
    # Handle edge cases like empty strings or invalid formats
    try:
        # Try various date formats
        date = datetime.strptime(cleaned_string, '%Y-%m-%d')
        if date:
            return date
        date = datetime.strptime(cleaned_string, '%m-%d-%Y')
        if date:
            return date
        date = datetime.strptime(cleaned_string, '%m/%d/%Y')
    except ValueError:
        return None

def string_to_float(s: str):
    """
    Converts a string to a float after removing non-numeric characters.
    
    Args:
        s (str): The string to convert.
        
    Returns:
        float: The converted float value.
        None: If the conversion fails or no valid number can be extracted.
    """
    # If input is already a float, return it
    if isinstance(s, float):
        return s
        
    if not isinstance(s, str):
        return None
    
    # Use regex to keep only valid numeric characters, including '-' for negatives and '.' for decimals
    cleaned_string = re.sub(r"[^\d.-]+", "", s)
    
    # Handle edge cases like empty strings or invalid formats
    try:
        return float(cleaned_string)
    except ValueError:
        return None

def string_to_int(s: str):
    """
    Converts a string to an integer after removing non-numeric characters.
    
    Args:
        s (str): The string to convert.
        
    Returns:
        int: The converted integer value.
        None: If the conversion fails or no valid number can be extracted.
    """
    # If input is already an int, return it
    if isinstance(s, int):
        return s
        
    if not isinstance(s, str):
        return None
    
    # Use regex to keep only digits and '-' for negatives
    cleaned_string = re.sub(r"[^\d-]+", "", s)
    
    # Handle edge cases like empty strings or invalid formats
    try:
        return int(cleaned_string)
    except ValueError:
        return None

def llm_clean(s, model_name='holesize', model_version='0'):
    """
    Converts a string to desired final data form for various pre-trained 
    language models.

    Args:
        s (str): The string to convert.
        model_name (str): The specific pre-trained model to use.
        model_version (str): Version of the model to use.
        
    Returns:
        pred (float): The cleaned output from the model.
    """
    # If input is already a cleaned value (assuming it's a string), return it
    if not isinstance(s, float):
        return s
    
    # Check devices
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Load pre-trained model checkpoint
    checkpoint_path = get_checkpoint_path(model_name, model_version)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint['model_config']
    data_parameters = checkpoint['data_parameters']
    labels = checkpoint['dataset_labels']
    num_tokens = checkpoint['num_tokens']

    # Define model, loss function and optimizer
    model_encoder = Encoder(
        vocab_dim=num_tokens, # Normally vocab but here num of unicode characters
        sequence_dim=data_parameters['sequence_size'],
        embed_dim=model_config['emb_dim'],
        ff_dim=model_config['ff_dim'],
        num_heads=model_config['num_heads'],
        num_blocks=model_config['num_blocks'],
        dropout=model_config['dropout'],
        norm_eps=model_config['layer_eps'],
        device=device
        ).to(device)

    # Define classifier
    model_classifier = Classifier(
        model_encoder, 
        sequence_dim=data_parameters['sequence_size'], 
        embedding_dim=model_config['emb_dim'], 
        num_classes=len(labels)
        ).to(device)
    
    # Load pre-trained model weights
    model_classifier.load_state_dict(checkpoint['model_state_dict'])

    # Encode the input
    dataset = HoleSize(
        None, labels, max_length=data_parameters['sequence_size']
        )
    
    # Tokenize the input
    X = np.array(dataset.tokenize(s)).reshape((1, -1))
    # print(X)

    model_classifier.eval()
    with torch.no_grad():
        y_pred = model_classifier(torch.tensor(X).to(device)).argmax(-1).to('cpu').numpy()[0]
    
    return dataset.classes[y_pred]

def clean_date(date_str: str) -> datetime | None:
    """
    Clean and standardize date strings into datetime objects.
    
    Args:
        date_str: String containing a date
        
    Returns:
        datetime object if successful, None if invalid date
    """
    # If input is already a datetime object, return it
    if isinstance(date_str, datetime):
        return date_str
        
    if not date_str or date_str in ['N/A', 'illegible', '-', 'BEFORE', 'SAME AS BEFORE']:
        return None
        
    # Remove any extra whitespace
    date_str = date_str.strip()
    
    # List of formats to try, in order of most to least common
    formats = [
        '%Y/%m/%d',           # 2020/8/1
        '%Y-%m-%d',           # 1973-02-29
        '%m/%d/%Y',           # 7/30/1971
        '%m/%d/%y',           # 11/29/54
        '%m-%d-%Y',           # 10-17-1983
        '%m-%d-%y',           # 5-27-66
        '%d-%b-%y',           # 29-Jul-71
        '%b %d, %Y',          # March 30. 1963
        '%B %d,%Y',           # April 28,1958
        '%b. %d, %Y',         # Sept. 11, 1957
        '%d/%m/%Y',           # 17/18/95 (ambiguous, assumes DD/MM/YYYY)
        '%Y'                  # 1949
    ]
    
    # Clean up some common variations
    date_str = re.sub(r'\.(?=\s|$)', '', date_str)  # Remove trailing periods
    date_str = re.sub(r'\s+', ' ', date_str)        # Normalize spaces
    
    # Try each format
    for fmt in formats:
        try:
            date = datetime.strptime(date_str, fmt)
            
            # Handle two-digit years - assume years > 50 are in the 1900s, <= 50 are in the 2000s
            if fmt.endswith('%y'):
                year = date.year
                if year >= 2050:  # If year is 2050 or later (from parsing '50' to '99')
                    # Adjust to 1950-1999
                    date = date.replace(year=year-100)
            
            # Convert to common format
            date_str = date.strftime('%m/%d/%Y')
            return date_str
        except ValueError:
            continue
            
    # Handle special cases
    if re.match(r'^\d{3,4}$', date_str):  # Handle year-only entries
        try:
            year = int(date_str)
            if 1900 <= year <= 2100:  # Reasonable year range
                return datetime(year, 1, 1).strftime('%m/%d/%Y')
        except ValueError:
            pass
            
    return None

def clean_bool(checkbox_str: str):
    '''
    check if string is valid representation of boolean
    
    args:
        checkbox_str: string of checkbox field
    returns:
        boolean: True if the input represents a positive/checked value, False otherwise
    '''
    # If input is already a boolean, return it
    if isinstance(checkbox_str, bool):
        return checkbox_str
        
    # If input is None or empty, return False
    if checkbox_str is None or (isinstance(checkbox_str, str) and not checkbox_str.strip()):
        return False
        
    try:
        # Normalize the string
        checkbox_str = checkbox_str.strip().upper()
        
        # Values that should be interpreted as True
        true_values = ['X', 'YES', 'TRUE', 'T', 'Y', '1']
        
        # Check for True values
        if checkbox_str in true_values:
            return True
        
        # Check for checkbox symbols
        if len(checkbox_str) == 1:
            if ord(checkbox_str) == 9745:  # Checked box symbol
                return True
            elif ord(checkbox_str) == 9744:  # Unchecked box symbol
                return False
        
        # All other values are considered False
        return False
    except AttributeError:
        # If it's not a string and not a boolean, return False
        return False

def convert_hole_size_to_decimal(size_str: str) -> float:
    """
    Converts oil/gas well hole size strings to decimal values.
    
    Common valid hole sizes are in 1/8 inch increments. Based on the dataset,
    the following fractions are allowed:
    
    Eighths (most common):
    - 1/8, 3/8, 5/8, 7/8
    
    Quarters:
    - 1/4, 3/4
    
    Halves:
    - 1/2
    
    Args:
        size_str: String containing the hole size (e.g. "8 3/4", "7-7/8", "13 3/8")
        
    Returns:
        Float decimal equivalent of the hole size or None if empty string
        
    Raises:
        ValueError: If the input string cannot be parsed or contains invalid fractions
    """
    # If input is already a float, return it
    if isinstance(size_str, float):
        return size_str
        
    # Check for empty string
    if not size_str or size_str.strip() == "":
        return None
        
    # Strip whitespace and quotes
    size_str = size_str.strip().strip('"').strip("'")
    
    # Handle pure decimal inputs
    try:
        return float(size_str)
    except ValueError:
        pass
        
    # Remove common separators
    size_str = size_str.replace('-', ' ').replace('/', ' ')
    
    # Split into whole and fractional parts
    parts = size_str.split()
    
    # Get the whole number
    try:
        whole = float(parts[0])
    except ValueError:
        raise ValueError(f"Invalid whole number: {parts[0]}")
        
    # If no fraction, return whole number
    if len(parts) == 1:
        return whole
        
    # Handle fraction part
    if len(parts) != 3:
        raise ValueError(f"Invalid fraction format: {size_str}")
        
    try:
        numerator = int(parts[1])
        denominator = int(parts[2])
    except ValueError:
        raise ValueError(f"Invalid fraction numbers: {parts[1]}/{parts[2]}")
        
    # Validate allowed fractions
    valid_fractions = {
        # Eighths
        (1,8): 0.125,
        (3,8): 0.375,
        (5,8): 0.625,
        (7,8): 0.875,
        
        # Quarters  
        (1,4): 0.25,
        (3,4): 0.75,
        
        # Halves
        (1,2): 0.5
    }
    
    fraction_tuple = (numerator, denominator)
    if fraction_tuple not in valid_fractions:
        raise ValueError(f"Invalid or uncommon fraction: {numerator}/{denominator}")
        
    return whole + valid_fractions[fraction_tuple]


if __name__ == '__main__':

    # LLM hole size cleaning
    input = '12-1/4'
    pred = llm_clean(input)
    print('Input: {}'.format(input))
    print('Cleaned hole size: {}\n'.format(pred))

    # Date cleaning
    input = '6/25/1971'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))

    input = '25/10/1971'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))

    input = '2020/8/1'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))

    input = 'April 28,1958'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date: {}\n'.format(date))
    
    # Test the new m-dd-yy format
    input = '5-27-66'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (m-dd-yy format): {}\n'.format(date))
    
    # Test more two-digit year formats
    input = '11/29/54'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (two-digit year 50s): {}\n'.format(date))
    
    input = '3-15-22'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (two-digit year 20s): {}\n'.format(date))
    
    input = '7/4/99'
    date = clean_date(input)
    print('Input: {}'.format(input))
    print('Cleaned date (two-digit year 90s): {}\n'.format(date))
    
    # Boolean cleaning
    input = ' yes '
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'true'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 't'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'y'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = '1'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'no'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = 'test'
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = None
    checkbox = clean_bool(input)
    print('Input: {}'.format(input))
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    input = ''
    checkbox = clean_bool(input)
    print('Input: empty string')
    print('Cleaned boolean: {}\n'.format(checkbox))
    
    # Hole size cleaning
    input = "8 3/4"
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))

    input = "7-7/8"
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))

    input = "13 3/8"
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: {}'.format(input))
    print('Hole size: {}\n'.format(hole_size))

    input = None
    hole_size = convert_hole_size_to_decimal(input)
    # Test with empty string
    input = ""
    hole_size = convert_hole_size_to_decimal(input)
    print('Input: empty string')
    print('Hole size: {}\n'.format(hole_size))
    
    # Test cases for handling inputs that are already in the target data type
    print("Testing functions with inputs already in target data type:\n")
    
    # Test string_to_float with float input
    float_val = 123.45
    result = string_to_float(float_val)
    print(f"Input: {float_val} (float)")
    print(f"string_to_float result: {result}, same object: {result is float_val}\n")
    
    # Test string_to_int with int input
    int_val = 42
    result = string_to_int(int_val)
    print(f"Input: {int_val} (int)")
    print(f"string_to_int result: {result}, same object: {result is int_val}\n")
    
    # Test clean_date with datetime input
    date_val = datetime.now()
    result = clean_date(date_val)
    print(f"Input: {date_val} (datetime)")
    print(f"clean_date result: {result}, same object: {result is date_val}\n")
    
    # Test clean_bool with boolean input
    bool_val = True
    result = clean_bool(bool_val)
    print(f"Input: {bool_val} (boolean)")
    print(f"clean_bool result: {result}, same object: {result is bool_val}\n")
    
    # Test convert_hole_size_to_decimal with float input
    float_val = 8.75
    result = convert_hole_size_to_decimal(float_val)
    print(f"Input: {float_val} (float)")
    print(f"convert_hole_size_to_decimal result: {result}, same object: {result is float_val}\n")
