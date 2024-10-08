import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        num_embeddings = len(embeddings) 
        dim = len(embeddings[0])
        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size
        # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze = True)
        # EX4
        


        # Aris' code   1ο Ερώτημα
        """self.linear = nn.Linear(2 * dim, 2*dim)
        self.relu = nn.ReLU()  
        self.output = nn.Linear(2 * dim, output_size)
        """


        #Kostis' code 
        # 4 - define a non-linear transformation of the representations
        self.linear = nn.Linear(dim, 1000)
        self.relu = nn.ReLU()  # EX5
        

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output = nn.Linear(1000, output_size)  # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        
        #Code for LabPrep 
        
        # 1 - embed the words, using the embedding layer
        embeddings = self.embeddings(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.sum(embeddings, dim=1)
        for i in range(lengths.shape[0]): # necessary to skip zeros in mean calculation
            representations[i] = representations[i] / lengths[i]  # EX6

        # 3 - transform the representations to new ones.
        representations = self.relu(self.linear(representations)) # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output(representations)  # EX6

        return logits


        # Code for main part 1ο Ερώτημα
        
        # embed the words, using the embedding layer
        embeddings = self.embeddings(x) 
        
        # Compute mean pooling
        mean_pooling = torch.mean(embeddings, dim=1)

        # Compute max pooling
        max_pooling, _ = torch.max(embeddings, dim=1)

        # Concatenate mean pooling and max pooling
        representations = torch.cat((mean_pooling, max_pooling), dim=1)

        # Transform the representations
        representations = self.relu(self.linear(representations))

        logits = self.output(representations)
        
        return logits



class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim=self.hidden_size)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        

        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        
        # pick the output of the LSTM corresponding to the last word
        
        representations = []
        for i, length in enumerate(lengths):
        # Calculate the index of the last word in the sequence
            last_index = length - 1 if length <= max_length else max_length - 1
            # Get the hidden state corresponding to the last word and append to list
            representations.append(ht[i, last_index, :])

        # Stack the representations into a single tensor
        representations = torch.stack(representations)
        


        logits = self.linear(representations)
        
        """
        representations = torch.zeros(batch_size, self.representation_size).float()
        for i in range(lengths.shape[0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]

        logits = self.linear(representations)
        """
        return logits
