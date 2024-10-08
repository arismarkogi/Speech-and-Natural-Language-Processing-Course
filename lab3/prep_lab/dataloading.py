import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import nltk 
nltk.download('punkt')

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        self.data = X
        self.labels = y
        self.word2idx = word2idx
        self.tokenized_data = [nltk.word_tokenize(sentence) for sentence in self.data]
        self.max_length = 15
        print("EX2: 10 Tokenized data examples:")
        for i in range(10):
            print(self.tokenized_data[i])
        # EX2

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
            """
        
        # Tokenize sentence
        example = self.tokenized_data[index]

        # Convert words to indices
        indexed_example = [self.word2idx[word] if word in self.word2idx else self.word2idx["<unk>"] for word in example]

        # Pad or truncate to max_length
        if len(indexed_example) < self.max_length:
            indexed_example += [0] * (self.max_length - len(indexed_example))
        else:
            indexed_example = indexed_example[:self.max_length]

        # Convert to tensor
        example_tensor = torch.tensor(indexed_example)

        # Get label
        label = self.labels[index]

        # Get length
        length  = len([x for x in example_tensor if x != 0])

        return example_tensor, label, length