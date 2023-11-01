from utils import Vectorizer
from networks import FFN
import torch
import numpy as np
from  typing import Dict
from utils import LMDataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


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
        # Definimos la función de pérdida
        self.loss_func = torch.nn.CrossEntropyLoss()
        # Definimos el optimizer
        self.optimizer = torch.optim.Adam(self.FFN.parameters())
        self.model_folder = Path.cwd() / Path('..').resolve() / Path('models', self.name)
        self.model_folder.mkdir(parents=True, exist_ok=True)

    def probabilities(self, contexts:list) -> float:
        '''
        Returns the estimated probabilities given a context.
        Input:
            - context_, list of words
        Output:
            - probabilities (tensor) according to model
        '''
        # Context to one-hot
        coded_context = self.code_context(contexts)
        # Feed network to obtain probabilities
        probabilities = self.FFN(coded_context)
        return probabilities

    def probability(self, words:list, contexts:list) -> float:
        '''
        Returns the probability of a word w given a context.
        Input:
            - word, string
            - context_, list of words
        Output:
            - probability (float) according to model
        '''
        # Context to one-hot
        coded_context = self.code_context(contexts)
        # Feed network to obtain probabilities
        probabilities = self.FFN(coded_context)
        # print(probabilities, probabilities.shape)
        if len(coded_context) == 1:
            idx = self.vectorizer.token_to_index(words)
            # print(idx)
            return probabilities[0][idx].item()
        else:
            indices = [self.vectorizer.token_to_index(word) for word in words]
            probabilities = [probabilities[i][idx] for i, idx in enumerate(indices)]
            return probabilities

    def perplexity(self, text):
        '''
        Returns the perplexity of the text
        according to the probabilities of the model.
        '''
        ds = LMDataset(texto=text, window_length=self.window_length)
        ds_loader = DataLoader(ds, batch_size=1, shuffle=False)
        probs = []
        for context, next_word in ds_loader:
            # Reconfiguramos los features
            context = [x[0] for x in context]
            # Reconfiguramos los targets
            next_word = list(next_word)
            print(f'context: {context}')
            print(f'next_word: {next_word}')
            prob = self.probability(next_word, context)
            print(f'probability: {prob}')
            probs.append(prob)
        n = len(probs)
        log_perplexity = -1/n * np.sum(np.log(probs))
        return np.exp(log_perplexity)


    def code_context(self, contexts):
        # Checking batched context
        shape_context = np.array(contexts).shape
        if len(shape_context) == 1:
            coded_context = self._get_coded_context(contexts)
        else:
            coded_context = [self._get_coded_context(context_).squeeze() for context_ in contexts]
            coded_context = torch.stack(coded_context)
        return coded_context

    def _get_coded_context(self, context):            
        # Context needs to be of window_length
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

    def train(self, texto:str, parametros:Dict[str, int]) -> None:
        '''
        Entrenamos la red sobre un texto usando unos parametros dados.
        '''
        # Instanciamos los parámetros
        self.optimizer.lr = parametros["learning_rate"]
        window_length = parametros["window_length"]
        batch_size = parametros["batch_size"]
        num_epochs = parametros["num_epochs"]
        running_loss = 0
        ds = LMDataset(texto=texto, window_length=window_length)
        for epoch in tqdm(range(num_epochs)):
            ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            # Iteramos sobre los batches
            batch_index = -1
            for ds_features, ds_labels in ds_loader:
                batch_index += 1
                # Reconfiguramos los features
                batch_len = len(ds_features[0])
                ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
                # Reconfiguramos los targets
                ds_labels = list(ds_labels)
                Y = torch.Tensor(self.vectorizer.token_to_index(ds_labels)).to(torch.int64)
                # the training routine is these 5 steps:
                # step 1. zero the gradients
                self.optimizer.zero_grad()
                # step 2. compute the output
                Y_hat = self.probabilities(ds_features)
                # step 3. compute the loss
                loss = self.loss_func(Y_hat, Y)
                loss_batch = loss.item()
                running_loss += (loss_batch - running_loss) / (batch_index + 1)
                # print(loss_batch, running_loss)
                # step 4. use loss to produce gradients
                loss.backward()
                # step 5. use optimizer to take gradient step
                self.optimizer.step()

    def save_model(self):
        torch.save(self.FFN, Path(self.model_folder, 'model.pth'))

    def load_model(self):
        self.FFN = torch.load(Path(self.model_folder, 'model.pth'))




