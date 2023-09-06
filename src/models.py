from utils import Vectorizer
from networks import FFN
import torch

class FFNLM :
    '''
    Defines a Feed-Forward Neural Network language model
        Args:
            vectorizer (Vocabulary): object to find one-hot encodings
            window_length (int): the length of the window of words
            embedding_dim (int): the output size of the word embedding
            hidden_size (int): the output size of the first Linear layer
            batch_first (bool): whether the 0th dimension is batch
    '''
    def __init__(self, vectorizer:Vectorizer, window_length:int, hidden_size:int):
        self.vectorizer = vectorizer
        self.window_length = window_length
        self.vocabulary_size = len(vectorizer)
        self.hidden_size = hidden_size
        self.FFN = FFN(window_length=self.window_length,\
                       vocabulary_size=self.vocabulary_size,\
                       hidden_size=self.hidden_size)
        self.name = 'ffn'

    def probability(self, word:str, context:list) -> float:
        '''
        Returns the probability of a word w given a context.
        Input:
            - word, string
            - context_, list of words
        Output:
            - probability (float) according to model
        '''
        # Context needs to be of window_length
        wl = self.window_length
        if len(context) > wl:
            context_ = context[-wl:]
        elif len(context) < wl:
            context_ = ['<BEGIN>']*(wl - len(context)) + context
        else:
            context_ = context
        assert(len(context_) == wl), f'\n\n{context_}\n\n{wl}'
        # Vector of word indexes
        coded_context = self.vectorizer.one_hot(context_)
        print('Context:', context_)
        print('Coded context:', coded_context)
        # context_coded = torch.stack([torch.tensor(self.vectorizer.lookup_token(w)) for w in context])
        # context_coded = torch.unsqueeze(context_coded, dim=0)
        # # Feed network to obtain probabilities
        # probabilities = self.FFN(context_coded)
        # probabilities = torch.squeeze(probabilities)
        # idx = self.vectorizer.lookup_token(word)
        # return probabilities[idx].item()

    def probabilities_outp(self, context:list) -> float:
        '''
        Returns a distribution probability over words given a context.
        Input:
            - context_, list of words
        Output:
            - list of probabilities for each token according to model
        '''
        # Context needs to be of window_length
        wl = self.FFN.window_length
        if len(context) > wl:
            context_ = context[-wl:]
        elif len(context) < wl:
            context_ = ['<BEGIN>']*(wl - len(context)) + context
        else:
            context_ = context
        assert(len(context_) == wl), f'\n\n{context_}\n\n{wl}'
        # Vector of word indexes
        context_coded = torch.stack([torch.tensor(self.vectorizer.lookup_token(w)) for w in context_])
        context_coded = torch.unsqueeze(context_coded, dim=0)
        # Feed network to obtain probabilities
        probabilities = self.FFN(context_coded)
        probabilities = torch.squeeze(probabilities)
        return probabilities.detach().numpy() 