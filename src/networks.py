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
        # ----------------------------------------------------------
        # Intermediate layer 1
        self.fc_intermediate_1 = nn.Linear(hidden_size, hidden_size)
        self.activation_fci = nn.functional.log_softmax
        # ----------------------------------------------------------
        # Intermediate layer 2
        self.fc_intermediate_2 = nn.Linear(hidden_size, hidden_size)
        self.activation_fci = nn.functional.log_softmax
        # ----------------------------------------------------------
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
        # # Find the input shape
        # batch_size, n_x = x_in.size()
           
        # Propagate to hidden layer
        x_in = x_in.to(torch.float32)
        # print(x_in.shape, x_in.dtype, self.fc1.weight.dtype)
        x_out = self.fc1(x_in)
        x_out = self.activation_fc1(x_out)

        # ----------------------------------------------------------
        # Propagate to intermediate layer 1
        x_intermediate = self.fc_intermediate_1(x_out)
        x_intermediate = self.activation_fci(x_intermediate, dim=1)
        # ----------------------------------------------------------
        # Propagate to intermediate layer 2
        x_intermediate = self.fc_intermediate_2(x_out)
        x_intermediate = self.activation_fci(x_intermediate, dim=1)
        # ----------------------------------------------------------
    
        # Propagate to output layer
        y = self.fc2(x_intermediate)
        y = self.activation_fc2(y, dim=1)

        return y

