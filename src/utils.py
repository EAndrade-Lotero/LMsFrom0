from typing import Union
import torch
from torchtext.vocab import build_vocab_from_iterator
import torch.nn.functional as F
import stanza


class Vectorizer :
    '''
    Clase para vectorizar texto.
    Input:
        - texto, lista de strings.
    '''

    def __init__(self, texto:Union[str, list]) -> None:
        tokens = self.get_tokens(texto)
        self.vocabulary = build_vocab_from_iterator(tokens)
        self.tokens = self.vocabulary.get_itos()
        print('tokens: ', self.tokens)

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
        nlp = stanza.Pipeline(lang='es', processors='tokenize', use_gpu=True, download_method=None)
        if isinstance(texto, list):
            texto = ' '.join(texto)
            doc = nlp(texto)
            return [[token.text.lower() for token in sentence.tokens] for sentence in doc.sentences]
        elif isinstance(texto, str):
            doc = nlp(texto)
            lista_listas = [[token.text.lower() for token in sentence.tokens] for sentence in doc.sentences]
            return lista_listas[0]
        else:
            print('OOOOOps!, tipo no aceptado', type(texto))
            raise Exception

    def token_one_hot(self, keys:Union[str, list]):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        if isinstance(keys, str):
            keys = [keys]
        return F.one_hot(torch.tensor(self.vocabulary(keys)), num_classes=len(self.vocabulary))

    def from_token_one_hot(self, tensors: torch.Tensor):
        '''
        ¿Qué hace esta función?

        Input:
            - ????

        Output:
            - ????
        '''
        indices = [torch.where(tensor == 1)[0].item() for tensor in tensors]
        palabras = self.vocabulary.lookup_tokens(indices)
        return palabras

    def one_hot(self, sentences: Union[str, list]):
        oraciones = self.get_tokens(sentences)
        if isinstance(sentences, list):
            one_hot_encoding = [self.token_one_hot(oracion) for oracion in oraciones] 
        if isinstance(sentences, str):
            one_hot_encoding = self.token_one_hot(oraciones)
        return one_hot_encoding