from typing import Union
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class HoleSize(Dataset):
    """
    A class to represent an oil and gas hole size dataset.

    Attributes:
        processed_text (list) : The cleaned text data represented as a list
            of observations. Each observation is a tuple where the first item is
            the review title and the second item is the review description.
        
        tokens (list) : The fully processed and tokenized dataset. A list of 
            tuples where the first element is the tokenized review title and the
            second item is the tokenized review description.
    """

    def __init__(self, path: Path, class_labels: dict, max_length: int=32):
        """
        Arguments:
            path (str) : A path like object to the data
            class_labels (dict) : A target class look up. For example casing 
                size of 8.625 (8 5/8") might map to class 2
            max_length (int)

        """

        # Class to label look up - i.e. 8.75 -> 1
        self.labels = class_labels
        # self.labels[-1] = 'PAD'
        # Label to class look up - i.e. 1 -> 8.75
        # Swaps from hole size: class id number to class id number: hole size
        self.classes = {v:k for k,v in self.labels.items()}
        self.max_length = max_length

        # Token 32 (first basic Latin token) becomes token 2
        self.padding_token = 0   # Overwrite unicode 0 to be the padding token
        self.unk_token = 1       # Overwrite unicode 1 to be unknown token
        self.token_shift = 30    # Do not use the first 32 unicode values
        self.num_tokens = 95 + 1 # Basic Latin codes plus padding token

        if path is not None:
            assert path.exists(), 'Path to the data must exist'

            # Read the raw data
            self.data = pd.read_csv(path, dtype={'raw':str, 'clean':float})

            # Tokenize the raw data
            self.data['tokens'] = self.data.apply(lambda x: self.tokenize(x.raw), axis=1)
            
            # Create class labels using the provided class labels
            self.data['target'] = self.data.apply(lambda x: self.labels[x.clean], axis=1)

            self.X = np.array(self.data['tokens'].to_list())
            self.Y = np.array(self.data['target'].to_list())

            # Define the number of observations
            self.num_obs = self.X.shape[0]

    def tokenize(self, s) -> Union[list[int]]:
        """
        Tokenize a complete dataset.

        Arguments:
            s (str) : The string to be tokenized.
        
        Returns:
            tokens (list) : The tokenized string, a list of integers.
        """

        # TODO: Handle unknown characters, no unicode value exists

        # Convert tokens to unicode number shifted by 31 so the first basic
        # Latin character (32 which is empty space) becomes 1
        tokens = []
        for c in s:
            if ord(c) < 32 or ord(c) > 126:
                tokens.append(self.unk_token)
            else:
                tokens.append(ord(c) - self.token_shift)

        # Pad the remainder of the sequence
        tokens.extend([self.padding_token]*(self.max_length - len(tokens)))

        return tokens

    def to_string(self, t) -> str:
        """
        Convert list of tokens to string

        Arguments:
            t (list) : The list of tokens to be converted
        
        Returns:
            string (str) : The string version of the token list
        """

        string = ''.join([chr(c + self.token_shift) for c in t if c !=self.padding_token])
        return string


    def __len__(self) -> int:
        return self.num_obs

    def __getitem__(self, idx) -> Union[tuple[np.ndarray]]:
        return self.X[idx], self.Y[idx]


class DataLoader:
    """
    A class to mimic the pytorch DataLoader but does not automatically convert
    the batch to a pytorch tensor
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle=False, seed=None, start_idx=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(len(dataset)/batch_size)) - start_idx
        self.shuffle = shuffle
        self.start_idx = start_idx

        assert self.start_idx == 0 or not shuffle
        if shuffle:
            self.rng = np.random.default_rng(seed)
        

    def __iter__(self):
        
        obs_indices = np.arange(len(self.dataset))

        if self.shuffle:
            self.rng.shuffle(obs_indices)

        for i in range(self.start_idx, self.num_batches):
            idx = obs_indices[i*self.batch_size:(i+1)*self.batch_size]
            yield self.dataset[idx]

def one_hot_encode(batch: Union[list[list]], sequence_size: int, vocab_size: int) -> np.ndarray:
    """
    Convert tokenized sequences into one hot encoded sequences.

    This method can be used to process a batch of tokenized sequences.

    Args:
        batch (list) : A batch of sequences where each sequence is
            represented by a list of word tokens.
        sequence_size (int) : The size of the sequences. All sequences in
            the batch must have this length.
        vocab_size (int) : The total number of tokens in vocabulary.

    Run example below to see how arrays can be used in advanced numpy 
    indexing.

    batch_size = 5
    vocab_size = 10
    sequence_size = 3

    batch_idx = np.repeat(np.arange(batch_size), sequence_size)
    sequence_idx = np.tile(np.arange(sequence_size), batch_size)
    vocab_idx = np.arange(vocab_size)

    tokens = np.array([0,1,2,5,9,1,0,5,2,3,4,5,9,8,7])
    ohe = np.zeros((batch_size, sequence_size, vocab_size))
    ohe[batch_idx, sequence_idx, tokens] = 1
    ohe
    """

    # The total number of observations
    batch_size = len(batch)
    
    # Create rank 3 tensor of 0's
    ohe = np.zeros((batch_size, sequence_size, vocab_size), dtype=np.int8)
    
    # Creates array repeating same obs idx for all words in sequence
    # 0,0,0,...,0, 1,1,1,...,1, 2,2,2,...,2 for n number of examples
    batch_idx = np.repeat(np.arange(batch_size), sequence_size)

    # Creates array that "iterates" through the sequences over and over
    # where s is the sequence dimension
    # 0,1,2,...,s, 0,1,2,...,s for n number of examples
    sequence_idx = np.tile(np.arange(sequence_size), batch_size)

    # Flatten the token list for indexing
    token_idx = np.array(batch).flatten()

    # OHE the tokens
    ohe[batch_idx, sequence_idx, token_idx] = 1

    return ohe

def positional_encoding(batch_size:int, sequence_size: int) -> np.ndarray:
    """
    Create OHE positional matrix

    This function returns a numpy array which can be used as input into a 
    positional embedding layer. The array at indices (-2, -1) inner most
    dimensions is the Identity matrix
    
    Arguments:
        batch_size (int)
        sequence_size (int) : The length of the sequences
    
    Returns:
        encoding (np.array) ; The positional encodings of shape (batch, 
        sequence, sequence)
    """

    # Identity matrix as single observation in a batch 
    # (1, sequence dim, sequence dim)
    identity = np.expand_dims(np.identity(sequence_size), 0)

    # Full batched version (batch dim, sequence dim, sequence dim)
    encoding = np.repeat(identity, batch_size, axis=0)

    return encoding
