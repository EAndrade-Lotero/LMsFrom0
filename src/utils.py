from typing import Union
import torch
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import stanza
from torch.utils.data import Dataset
import numpy as np

# stanza.download(lang='es')
dash_line = '-'*20

class Vectorizer :
    '''
    Clase para vectorizar texto.
    Input:
        - texto, lista de strings.
    '''

    def __init__(self, texto:Union[str, list]) -> None:
        flatten_tokens = self.get_tokens(texto)
        print(list(set(flatten_tokens)))
        self.vocabulary = build_vocab_from_iterator(flatten_tokens, specials=["<unk>", "<eos>", "<begin>"])
        self.vocabulary.set_default_index(self.vocabulary["<unk>"])
        self.tokens = self.vocabulary.get_itos()
        print(self.tokens)


    def __len__(self):
        return len(self.tokens)
    
    def get_tokens(self, texto):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        nlp = stanza.Pipeline(lang='es', 
                              processors='tokenize', 
                              use_gpu=False, 
                              download_method=None,
                              verbose=False)
        if isinstance(texto, list):
            for oracion in texto:
                doc = nlp(oracion)
                listas_tokens = [token.text.lower() for sentence in doc.sentences for token in sentence.tokens]
            return listas_tokens
        elif isinstance(texto, str):
            doc = nlp(texto)
            listas_tokens = [token.text.lower() for sentence in doc.sentences for token in sentence.tokens]
            return listas_tokens
        else:
            print('OOOOOps!, tipo no aceptado', type(texto))
            raise Exception

    def token_to_index(self, keys:Union[str, list]):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        if isinstance(keys, str):
            keys = [keys]
        return self.vocabulary(keys)

    def index_to_token(self, indices:list):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        return self.vocabulary.lookup_tokens(indices)

    def token_to_one_hot(self, keys:Union[str, list]):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        if isinstance(keys, str):
            keys = [keys]
        return F.one_hot(torch.tensor(self.token_to_index(keys)), num_classes=len(self.vocabulary))

    def one_hot_to_token(self, batch_tensors: torch.Tensor):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        lista_indices = []
        for tensors in batch_tensors:
            lista_indices.append([torch.where(tensor == 1)[0].item() for tensor in tensors])
        batch_palabras = []
        for indices in lista_indices:
            batch_palabras.append(self.index_to_token(indices))
        return batch_palabras

    def one_hot(self, sentences: Union[str, list], padding:Union[int, None]=None):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        def pad_sentence(sentence):
            len_sentence = len(sentence)
            if len_sentence >= max_len:
                return sentence[:max_len]
            padding_tokens = ['<eos>'] * (max_len - len_sentence)
            return sentence + padding_tokens

        if isinstance(sentences, list):
            oraciones = self.get_tokens(sentences)
            if padding is None:
                max_len = max([len(oracion) for oracion in oraciones])
            else:
                max_len = padding
            oraciones = [pad_sentence(oracion) for oracion in oraciones]
            one_hot_encoding = [self.token_to_one_hot(oracion) for oracion in oraciones] 
            one_hot_encoding = torch.stack(one_hot_encoding)
        if isinstance(sentences, str):
            oracion = self.get_tokens(sentences)
            max_len = padding if padding is not None else len(oracion)
            oracion = pad_sentence(oracion)
            one_hot_encoding = self.token_to_one_hot(oracion)
            # Convertimos a batch de tamaño 1
            one_hot_encoding = one_hot_encoding.unsqueeze(dim=0)
        return one_hot_encoding
    
class LMDataset(Dataset):
    '''
    Dataset que toma un texto y devuelve los pares
    X: contexto de tamaño k (ventana)
    Y: siguiente palabra
    '''
    def __init__(self, texto:str, vectorizer: Vectorizer, window_length:int=2) -> None: 
        vec = vectorizer
        lista_tokens = vec.get_tokens(texto)
        len_tokens = len(lista_tokens)
        X = []
        Y = []
        for i in range(len_tokens - window_length):
            ventana = lista_tokens[i:i+window_length]
            next_token = lista_tokens[i+window_length]
            X.append(ventana)
            Y.append(next_token)
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
  