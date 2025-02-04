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
        return datetime.strptime(cleaned_string, '%Y-%m-%d')
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


if __name__ == '__main__':
    pred = llm_clean('12-1/4')
    print('Cleaned hole size: {}'.format(pred))

    date = string_to_date('6/25/1971')
    print('Cleaned date: {}'.format(date))