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
        pred: The cleaned output from the model.
    """
    
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
            date = datetime.strptime(date_str, fmt).date()
            # Convert to common format
            date = date.strftime('%m/%d/%Y')
            return date
        except ValueError:
            continue
            
    # Handle special cases
    if re.match(r'^\d{3,4}$', date_str):  # Handle year-only entries
        try:
            year = int(date_str)
            if 1900 <= year <= 2100:  # Reasonable year range
                return datetime(year, 1, 1)
        except ValueError:
            pass
            
    return None

def clean_bool(checkbox_str: str):
    '''
    check if string is valid representation of boolean
    
    args:
        checkbox_str: string of checkbox field
    returns:
        boolean if successful or the string itself if unknown
    '''
    try:
        checkbox_str = checkbox_str.strip().upper()
        if (checkbox_str == 'X') or (checkbox_str == 'YES'):
            return True
        elif (checkbox_str == '') or (checkbox_str == 'NO'):
            return False
        elif len(checkbox_str) == 1: #check box symbol
            if ord(checkbox_str) == 9745:
                return True
            elif ord(checkbox_str) == 9744:
                return False
        else:
            print(checkbox_str,'is a unknown checkbox')
            return None
    except AttributeError:
        print(checkbox_str,'not a string')

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
        Float decimal equivalent of the hole size
        
    Raises:
        ValueError: If the input string cannot be parsed or contains invalid fractions
    """
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
    pred = llm_clean('12-1/4')
    print('Cleaned hole size: {}'.format(pred))

    # Date cleaning
    date = clean_date('6/25/1971')
    print('Cleaned date: {}'.format(date))

    date = clean_date('25/10/1971')
    print('Cleaned date: {}'.format(date))

    date = clean_date('2020/8/1')
    print('Cleaned date: {}'.format(date))

    date = clean_date('April 28,1958')
    print('Cleaned date: {}'.format(date))
    
    checkbox = clean_bool(' yes ')
    print(checkbox)
    
    checkbox = clean_bool('test')
    print(checkbox)
    
    checkbox = clean_bool(None)
    print(checkbox)
    
    # Hole size cleaning
    hole_size = convert_hole_size_to_decimal("8 3/4") # 8.75
    print('Hole size: {}'.format(hole_size))

    hole_size = convert_hole_size_to_decimal("7-7/8") # 7.875
    print('Hole size: {}'.format(hole_size))

    hole_size = convert_hole_size_to_decimal("13 3/8") # 13.375
    print('Hole size: {}'.format(hole_size))
