import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from  typing import Dict, Optional
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from utils.utils import LMDataset
from utils.utils import Vectorizer
from lms.networks import FFN, ZeroLayerTransformer


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
    
    def summary(self):
        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in self.FFN.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f'Total Trainable Params: {total_params}')
        if next(self.FFN.parameters()).is_cuda:
            print('Model device: cuda')
        elif next(self.FFN.parameters()).is_mps:
            print('Model device: mps')
        else:
            print('Model device: cpu')
        

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
        #print('HOLA', len(coded_context.shape))
        if len(coded_context.shape) == 1:
            idx = self.vectorizer.token_to_index(words)
            #print(idx)
            return probabilities[idx].item()
        else:
            indices = [self.vectorizer.token_to_index(word) for word in words]
            probabilities = [probabilities[i][idx] for i, idx in enumerate(indices)]
            return probabilities

    def perplexity(self, text):
        '''
        Returns the perplexity of the text
        according to the probabilities of the model.
        '''
        ds = LMDataset(texto=text, vectorizer=self.vectorizer, window_length=self.window_length)
        ds_loader = DataLoader(ds, batch_size=1, shuffle=False)
        probs = []
        for context, next_word in ds_loader:
            # Reconfiguramos los features
            context = [x[0] for x in context]
            # Reconfiguramos los targets
            next_word = list(next_word)
            #print(f'context: {context}')
            #print(f'next_word: {next_word}')
            prob = self.probability(next_word, context)
            #print(f'probability: {prob}')
            probs.append(prob)
        n = len(probs)
        log_perplexity = -1/n * np.sum(np.log(probs))
        return np.exp(log_perplexity)


    def code_context(self, contexts):
        # Checking batched context
        shape_context = np.array(contexts).shape
        #print('shape_context: ',shape_context)
        if len(shape_context) == 1:
            coded_context = self._get_coded_context(contexts)
        else:
            coded_context = [self._get_coded_context(context_).squeeze() for context_ in contexts]
            shapes = [t.shape for t in coded_context]
            #print('Shapes:', shapes)
            coded_context = torch.stack(coded_context)
        return coded_context

    def _get_coded_context(self, context):            
        # Context needs to be of window_length
        wl = self.window_length
        if len(context) > wl:
            context_ = context[-wl:]
        elif len(context) < wl:
            context_ = ['<sos>']*(wl - len(context)) + context
        else:
            context_ = context
        assert(len(context_) == wl), f'\n\n{context_}\n\n{wl}'
        # Vector of word indexes
        #context_ = ' '.join(context_)
        #print('context_: ',context_)
        one_hot_context = self.vectorizer.tokens_to_one_hot(context_)
        #print('before flatten: ',one_hot_context)
        #print(one_hot_context.shape)
        one_hot_context = torch.flatten(one_hot_context, start_dim=0)
        #print('after flatten: ',one_hot_context)
        #print(one_hot_context.shape)
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
        loss_history = []  # Lista para almacenar la pérdida de cada época
        
        ds = LMDataset(texto=texto, vectorizer=self.vectorizer, window_length=window_length)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for epoch in tqdm(range(num_epochs)):
            # Iteramos sobre los batches
            batch_index = -1
            for ds_features, ds_labels in ds_loader:
                batch_index += 1
                #if batch_index<202:
                #    continue
                # Verifica la longitud de ds_features y ds_labels
                # if len(ds_features) != len(ds_labels):
                #     print(f"Error: Las longitudes no coinciden. ds_features: {len(ds_features)}, ds_labels: {len(ds_labels)}")
                #     continue
                # Reconfiguramos los features
                batch_len = len(ds_features[0])
                ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
                #print('batch_index: ', batch_index, 'ds_features: ',ds_features)
                #print('porcentaje_batch_index: ',round(100*batch_index/(len(ds)/batch_size),2),'%')
                # Reconfiguramos los targets
                ds_labels = list(ds_labels)
                Y = torch.Tensor(self.vectorizer.token_to_index(ds_labels)).to(torch.int64).to(self.FFN.device)
                # the training routine is these 5 steps:
                # step 1. zero the gradients
                self.optimizer.zero_grad()
                # step 2. compute the output
                Y_hat = self.probabilities(ds_features).to(self.FFN.device)
                # step 3. compute the loss
                loss = self.loss_func(Y_hat, Y)
                loss_batch = loss.item()
                running_loss += (loss_batch - running_loss) / (batch_index + 1)
                # print(loss_batch, running_loss)
                # step 4. use loss to produce gradients
                loss.backward()
                # step 5. use optimizer to take gradient step
                self.optimizer.step()
                # # Guardamos la pérdida de la época
                loss_history.append(loss_batch)
        
        # # Graficamos la pérdida con Seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(loss_history)), y=loss_history, label='Pérdida de entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Evolución de la pérdida durante el entrenamiento')
        plt.legend()
        plt.show()

    def save_model(self):
        torch.save(self.FFN, Path(self.model_folder, 'model.pth'))

    def load_model(self):
        self.FFN = torch.load(Path(self.model_folder, 'model.pth'))

    def next_word(self, context:list) -> str:
        '''
        Picks a word following context according to
        model's probabilities.
        Input:
            - context, list of words
        '''
        word_probs = self.probabilities(context).detach().numpy() 
        suma = sum(word_probs)
        #print(suma)
        if 0 < abs(1 - suma) < 1e-5:
            word_probs = [x/suma for x in word_probs]
        try:
            indice = np.random.choice(self.vectorizer.tokens, size=1, p=word_probs)[0]
            #print(indice)
            return indice
        except Exception as e:
            print('Ay mk!')
            raise Exception(e)

    def continue_context(self, context:list, max_words:int=10) -> list:
        '''
        Takes a context and continues generation of next
        word until <eos> occurs or max_words are generated.
        '''
        generated = []
        contador = 0
        context = [w for w in context]
        #print(' '.join(context), end=' ')
        for _ in range(max_words):
            word = self.next_word(context)
            #print(word, end=' ')
            if (word == '<eos>'):
                break
            generated.append(word)
            context.append(word)
            contador += 1
        return generated


class ZLT :
    '''
    Defines a Zero Layer Transformer language model
        Args:
            vectorizer (Vocabulary): object to find one-hot encodings
            window_length (int): the length of the window of words
            embedding_dim (int): the output size of the word embedding
            batch_first (bool): whether the 0th dimension is batch
    '''
    def __init__(self, vectorizer:Vectorizer, window_length:int):
        self.vectorizer = vectorizer
        self.window_length = window_length
        self.vocabulary_size = len(vectorizer)
    
        self.ZLT = ZeroLayerTransformer(window_length=self.window_length,\
                       vocabulary_size=self.vocabulary_size)
        self.name = 'zlt'
        # Definimos la función de pérdida
        self.loss_func = torch.nn.CrossEntropyLoss()
        # Definimos el optimizer
        self.optimizer = torch.optim.Adam(self.ZLT.parameters())
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
        probabilities = self.ZLT(coded_context)
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
        probabilities = self.ZLT(coded_context)
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
        ds = LMDataset(texto=text, vectorizer=self.vectorizer, window_length=self.window_length)
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
        #print('shape_context: ',shape_context)
        if len(shape_context) == 1:
            coded_context = self._get_coded_context(contexts)
        else:
            coded_context = [self._get_coded_context(context_).squeeze() for context_ in contexts]
            shapes = [t.shape for t in coded_context]
            #print('Shapes:', shapes)
            coded_context = torch.stack(coded_context)
        return coded_context

    def _get_coded_context(self, context):            
        # Context needs to be of window_length
        wl = self.window_length
        if len(context) > wl:
            context_ = context[-wl:]
        elif len(context) < wl:
            context_ = ['<sos>']*(wl - len(context)) + context
        else:
            context_ = context
        assert(len(context_) == wl), f'\n\n{context_}\n\n{wl}'
        # Vector of word indexes
        #context_ = ' '.join(context_)
        #print('context_: ',context_)
        one_hot_context = self.vectorizer.tokens_to_one_hot(context_)
        #print('before flatten: ',one_hot_context)
        #print(one_hot_context.shape)
        one_hot_context = torch.flatten(one_hot_context, start_dim=0)
        #print('after flatten: ',one_hot_context)
        #print(one_hot_context.shape)
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
        ds = LMDataset(texto=texto, vectorizer=self.vectorizer, window_length=window_length)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in tqdm(range(num_epochs)):
            # Iteramos sobre los batches
            batch_index = -1
            for ds_features, ds_labels in ds_loader:
                batch_index += 1
                #if batch_index<202:
                #    continue
                # Reconfiguramos los features
                batch_len = len(ds_features[0])
                ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
                #print('batch_index: ', batch_index, 'ds_features: ',ds_features)
                #print('porcentaje_batch_index: ',round(100*batch_index/(len(ds)/batch_size),2),'%')
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
        torch.save(self.ZLT, Path(self.model_folder, 'model.pth'))

    def load_model(self):
        self.ZLT = torch.load(Path(self.model_folder, 'model.pth'))


class Transformer :
    '''
    Defines a Transformer Network language model
        Args:
            vectorizer (Vocabulary): object to find one-hot encodings
            d_model (int): the number of expected features in the encoder/decoder inputs (default=512).
            nhead (int): the number of heads in the multiheadattention models (default=8).
            num_encoder_layers (int): the number of sub-encoder-layers in the encoder (default=6).
            num_decoder_layers (int): the number of sub-decoder-layers in the decoder (default=6).
            dim_feedforward (int): the dimension of the feedforward network model (default=2048).
            dropout (float): the dropout value (default=0.1).
            norm_first (bool): if True, encoder and decoder layers will perform LayerNorms before other attention and feedforward operations, otherwise after. Default: False (after).
            bias (bool): If set to False, Linear and LayerNorm layers will not learn an additive bias. Default: True.
            batch_first (bool): whether the 0th dimension is batch
    '''
    def __init__(
                self, 
                vectorizer: Vectorizer, 
                d_model: int,
                nhead: int,
                num_encoder_layers: int,
                num_decoder_laters: int,
                dim_feedforward: int,
                dropout: Optional[float]=0.1,
                norm_first: Optional[bool]=False,
                bias: Optional[bool]=True,
                batch_first: Optional[bool]=True
            ) -> None:
        self.vectorizer = vectorizer
        self.FFN = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_laters,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            bias=bias,
            batch_first=batch_first
        )
        self.name = 'transformer'
        # Definimos la función de pérdida
        self.loss_func = torch.nn.CrossEntropyLoss()
        # Definimos el optimizer
        self.optimizer = torch.optim.Adam(self.FFN.parameters())
        self.model_folder = Path.cwd() / Path('..').resolve() / Path('models', self.name)
        self.model_folder.mkdir(parents=True, exist_ok=True)
    
    def summary(self):
        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in self.FFN.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f'Total Trainable Params: {total_params}')
        if next(self.FFN.parameters()).is_cuda:
            print('Model device: cuda')
        elif next(self.FFN.parameters()).is_mps:
            print('Model device: mps')
        else:
            print('Model device: cpu')
        

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
        #print('HOLA', len(coded_context.shape))
        if len(coded_context.shape) == 1:
            idx = self.vectorizer.token_to_index(words)
            #print(idx)
            return probabilities[idx].item()
        else:
            indices = [self.vectorizer.token_to_index(word) for word in words]
            probabilities = [probabilities[i][idx] for i, idx in enumerate(indices)]
            return probabilities

    def perplexity(self, text):
        '''
        Returns the perplexity of the text
        according to the probabilities of the model.
        '''
        ds = LMDataset(texto=text, vectorizer=self.vectorizer, window_length=self.window_length)
        ds_loader = DataLoader(ds, batch_size=1, shuffle=False)
        probs = []
        for context, next_word in ds_loader:
            # Reconfiguramos los features
            context = [x[0] for x in context]
            # Reconfiguramos los targets
            next_word = list(next_word)
            #print(f'context: {context}')
            #print(f'next_word: {next_word}')
            prob = self.probability(next_word, context)
            #print(f'probability: {prob}')
            probs.append(prob)
        n = len(probs)
        log_perplexity = -1/n * np.sum(np.log(probs))
        return np.exp(log_perplexity)


    def code_context(self, contexts):
        # Checking batched context
        shape_context = np.array(contexts).shape
        #print('shape_context: ',shape_context)
        if len(shape_context) == 1:
            coded_context = self._get_coded_context(contexts)
        else:
            coded_context = [self._get_coded_context(context_).squeeze() for context_ in contexts]
            shapes = [t.shape for t in coded_context]
            #print('Shapes:', shapes)
            coded_context = torch.stack(coded_context)
        return coded_context

    def _get_coded_context(self, context):            
        # Context needs to be of window_length
        wl = self.window_length
        if len(context) > wl:
            context_ = context[-wl:]
        elif len(context) < wl:
            context_ = ['<sos>']*(wl - len(context)) + context
        else:
            context_ = context
        assert(len(context_) == wl), f'\n\n{context_}\n\n{wl}'
        # Vector of word indexes
        #context_ = ' '.join(context_)
        #print('context_: ',context_)
        one_hot_context = self.vectorizer.tokens_to_one_hot(context_)
        #print('before flatten: ',one_hot_context)
        #print(one_hot_context.shape)
        one_hot_context = torch.flatten(one_hot_context, start_dim=0)
        #print('after flatten: ',one_hot_context)
        #print(one_hot_context.shape)
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
        loss_history = []  # Lista para almacenar la pérdida de cada época
        
        ds = LMDataset(texto=texto, vectorizer=self.vectorizer, window_length=window_length)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for epoch in tqdm(range(num_epochs)):
            # Iteramos sobre los batches
            batch_index = -1
            for ds_features, ds_labels in ds_loader:
                batch_index += 1
                #if batch_index<202:
                #    continue
                # Verifica la longitud de ds_features y ds_labels
                # if len(ds_features) != len(ds_labels):
                #     print(f"Error: Las longitudes no coinciden. ds_features: {len(ds_features)}, ds_labels: {len(ds_labels)}")
                #     continue
                # Reconfiguramos los features
                batch_len = len(ds_features[0])
                ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
                #print('batch_index: ', batch_index, 'ds_features: ',ds_features)
                #print('porcentaje_batch_index: ',round(100*batch_index/(len(ds)/batch_size),2),'%')
                # Reconfiguramos los targets
                ds_labels = list(ds_labels)
                Y = torch.Tensor(self.vectorizer.token_to_index(ds_labels)).to(torch.int64).to(self.FFN.device)
                # the training routine is these 5 steps:
                # step 1. zero the gradients
                self.optimizer.zero_grad()
                # step 2. compute the output
                Y_hat = self.probabilities(ds_features).to(self.FFN.device)
                # step 3. compute the loss
                loss = self.loss_func(Y_hat, Y)
                loss_batch = loss.item()
                running_loss += (loss_batch - running_loss) / (batch_index + 1)
                # print(loss_batch, running_loss)
                # step 4. use loss to produce gradients
                loss.backward()
                # step 5. use optimizer to take gradient step
                self.optimizer.step()
                # # Guardamos la pérdida de la época
                loss_history.append(loss_batch)
        
        # # Graficamos la pérdida con Seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(loss_history)), y=loss_history, label='Pérdida de entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Evolución de la pérdida durante el entrenamiento')
        plt.legend()
        plt.show()

    def save_model(self):
        torch.save(self.FFN, Path(self.model_folder, 'model.pth'))

    def load_model(self):
        self.FFN = torch.load(Path(self.model_folder, 'model.pth'))

    def next_word(self, context:list) -> str:
        '''
        Picks a word following context according to
        model's probabilities.
        Input:
            - context, list of words
        '''
        word_probs = self.probabilities(context).detach().numpy() 
        suma = sum(word_probs)
        #print(suma)
        if 0 < abs(1 - suma) < 1e-5:
            word_probs = [x/suma for x in word_probs]
        try:
            indice = np.random.choice(self.vectorizer.tokens, size=1, p=word_probs)[0]
            #print(indice)
            return indice
        except Exception as e:
            print('Ay mk!')
            raise Exception(e)

    def continue_context(self, context:list, max_words:int=10) -> list:
        '''
        Takes a context and continues generation of next
        word until <eos> occurs or max_words are generated.
        '''
        generated = []
        contador = 0
        context = [w for w in context]
        #print(' '.join(context), end=' ')
        for _ in range(max_words):
            word = self.next_word(context)
            #print(word, end=' ')
            if (word == '<eos>'):
                break
            generated.append(word)
            context.append(word)
            contador += 1
        return generated