import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union

class FFN(nn.Module):
    """ A Multilayer Perceptron with one-hot encoding as first layer
        and a window of words of length window_length
    """
    def __init__(
                self, 
                window_length:int, 
                vocabulary_size:int, 
                hidden_size:list,
                embeddings_dim:Optional[Union[None, int]]=None
            ) -> None:
        """
        Args:
            vocabulary_size (int): the vocabulary size
            hidden_sizes (list): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
            batch_first (bool): whether the 0th dimension is batch
        """
        super(FFN, self).__init__()
        self.window_length = window_length  #k
        self.vocabulary_size = vocabulary_size  # |V|
        self.hidden_size = hidden_size

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        # -------------------------------------
        # Defining the layers
        # -------------------------------------
        # Hidden layer
        if embeddings_dim is None:
            embeddings_dim = vocabulary_size
        self.fc1 = nn.Linear(embeddings_dim * window_length, hidden_size).to(self.device)
        self.activation_fc1 = torch.sigmoid
        # ----------------------------------------------------------
        # Intermediate layer 1
        self.fc_intermediate_1 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.activation_fci = torch.sigmoid
        # ----------------------------------------------------------
        # Intermediate layer 2
        self.fc_intermediate_2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.activation_fci = torch.sigmoid
        # ----------------------------------------------------------
        # Output layer
        self.fc2 = nn.Linear(hidden_size, vocabulary_size).to(self.device)
        self.activation_fc2 = nn.functional.softmax

        # self.model = nn.Sequential(
        #     self.fc1,
        #     self.activation_fc1,
        #     self.fc_intermediate_1,
        #     self.activation_fci,
        #     self.fc_intermediate_2,
        #     self.activation_fci,
        #     self.fc2,
        #     self.activation_fc2
        # )

        # self.model.to(self.device)
        

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
        # print(x_in.shape)  
        # Propagate to hidden layer
        x_in = x_in.to(torch.float32).to(self.device)
        #y = self.model(x_in)

        # print(x_in.shape, x_in.dtype, self.fc1.weight.dtype)
        x_out = self.fc1(x_in)
        x_out = self.activation_fc1(x_out)

        # ----------------------------------------------------------
        # Propagate to intermediate layer 1
        x_intermediate = self.fc_intermediate_1(x_out)
        x_intermediate = self.activation_fci(x_intermediate)
        # ----------------------------------------------------------
        # Propagate to intermediate layer 2
        x_intermediate = self.fc_intermediate_2(x_out)
        x_intermediate = self.activation_fci(x_intermediate)
        # ----------------------------------------------------------
    
        # Propagate to output layer
        y = self.fc2(x_intermediate)
        y = self.activation_fc2(y, dim=-1)

        return y


class ZeroLayerTransformer(nn.Module):
    def __init__(self, window_length:int, vocabulary_size:int):
        """
        Args:
            vocabulary_size (int): the vocabulary size
            hidden_sizes (list): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
            batch_first (bool): whether the 0th dimension is batch
        """
        super(ZeroLayerTransformer, self).__init__()
        self.window_length = window_length  #k
        self.vocabulary_size = vocabulary_size  #|V|
        # -------------------------------------
        # Defining the layers
        # -------------------------------------
        # Activation layer
        self.fc1 = nn.Linear(vocabulary_size * window_length, vocabulary_size)
        self.activation_fc1 = nn.functional.softmax
        

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
        x_out = self.activation_fc1(x_out, dim=1)

        return x_out
    

