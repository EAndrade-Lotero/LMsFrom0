from utils import Vectorizer
from networks import FFN
import torch
import numpy as np

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

    def probability(self, words:list, contexts:list) -> float:
        '''
        Returns the probability of a word w given a context.
        Input:
            - word, string
            - context_, list of words
        Output:
            - probability (float) according to model
        '''
        # Checking batched context
        shape_context = np.array(contexts).shape
        if len(shape_context) == 1:
            coded_context = self._get_coded_context(contexts)
        else:
            coded_context = [self._get_coded_context(context_).squeeze() for context_ in contexts]
            coded_context = torch.stack(coded_context)
        print('Context:', contexts)
        print(self.vectorizer.tokens)
        print('Coded context:', coded_context)
        print(coded_context.shape)
        # Feed network to obtain probabilities
        probabilities = self.FFN(coded_context)
        print(probabilities, probabilities.shape)
        if len(shape_context) == 1:
            idx = self.vectorizer.token_to_index(words)
            print(idx)
            return probabilities[0][idx].item()
        else:
            indices = [self.vectorizer.token_to_index(word) for word in words]
            probabilities = [probabilities[i][idx] for i, idx in enumerate(indices)]
            return probabilities
   
    def _get_coded_context(self, context):            # Context needs to be of window_length
        wl = self.window_length
        if len(context) > wl:
            context_ = context[-wl:]
        elif len(context) < wl:
            context_ = ['<begin>']*(wl - len(context)) + context
        else:
            context_ = context
        assert(len(context_) == wl), f'\n\n{context_}\n\n{wl}'
        # Vector of word indexes
        context_ = ' '.join(context_)
        one_hot_context = self.vectorizer.one_hot(context_)
        # print(one_hot_context.shape)
        one_hot_context = torch.flatten(one_hot_context, start_dim=1, end_dim=2)
        # print(one_hot_context.shape)
        return one_hot_context
