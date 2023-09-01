from typing import Union, Iterable
import torch
import torch.nn as nn
import torchtext
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import stanza

stanza.download(lang='es')

def get_tokens(texto):
    '''
    ¿Qué hace esta función?

    Input:
        - ????

    Output:
        - ????
    '''
    nlp = stanza.Pipeline(lang='es', processors='tokenize', use_gpu=True, download_method=None)
    doc = nlp(texto)
    return [[token.text.lower() for token in sentence.tokens] for sentence in doc.sentences]

def my_one_hot(voc, keys:Union[str, Iterable]):
    '''
    ¿Qué hace esta función?

    Input:
        - ????

    Output:
        - ????
    '''
    if isinstance(keys, str):
        keys = [keys]
    return F.one_hot(torch.tensor(voc(keys)), num_classes=len(voc))


def from_one_hot(voc, tensors: torch.Tensor):
    '''
    ¿Qué hace esta función?

    Input:
        - ????

    Output:
        - ????
    '''
    indices = [torch.where(tensor == 1)[0].item() for tensor in tensors]
    palabras = voc.lookup_tokens(indices)
    return palabras

texto = '¡Hola mundo! Hola mundo no seas tan cruel'
tokens = get_tokens(texto)
print('tokens:', tokens)
voc = build_vocab_from_iterator(tokens)
x = voc.get_itos()
print('itos: ', x)

one_hots = my_one_hot(voc,['seas','mundo'])
print('one hots:', one_hots)
decoded = from_one_hot(voc, one_hots)
print(decoded)

