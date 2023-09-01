import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    """ A Multilayer Perceptron with one-hot encoding as first layer
        and a window of words of length window_length
    """
    def __init__(self, window_length:int, vocabulary_size:int, hidden_size:list):
        """
        Args:
            vocabulary_size (int): the vocabulary size
            hidden_sizes (list): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
            batch_first (bool): whether the 0th dimension is batch
        """
        super(FFN, self).__init__()
        self.window_length = window_length  #k
        self.vocabulary_size = vocabulary_size  #|V|
        self.hidden_size = hidden_size
        # -------------------------------------
        # Defining the layers
        # -------------------------------------
        # Hidden layer
        self.fc1 = nn.Linear(vocabulary_size * window_length, hidden_size)
        self.activation_fc1 = nn.functional.sigmoid
        # Output layer
        self.fc2 = nn.Linear(hidden_size, vocabulary_size)
        self.activation_fc2 = nn.functional.softmax

    def forward(self, x_in):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch, vocabulary_size)
        """
        # Find the input shape
        batch_size, n_x = x_in.size()
           
        # Propagate to hidden layer
        x_out = self.fc1(x_in)
        x_out = self.activation_fc1(x_out)
        
        # Propagate to output layer
        y = self.fc2(x_out)
        y = self.activation_fc2(y)
        return y

