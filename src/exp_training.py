from utils import LMDataset
import torch
from torch.utils.data import DataLoader
from utils import Vectorizer
from models import FFNLM

def test_cross_entropy():

    texto = ['Ana Beto Carlos David']
    vec = Vectorizer(texto)
    window_length = 2
    hidden_size = 10
    batch_size = 2
    parameters = {"vectorizer":vec,
                "window_length": window_length,
                "hidden_size":hidden_size,
    }
    lm = FFNLM(**parameters)
    ds = LMDataset(texto=texto, window_length=window_length)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    context, next_word = next(iter(ds_loader))
    context = [[x[i] for x in context] for i in range(batch_size)]
    next_word = list(next_word)
    print(f'contexto:{context}, siguiente:{next_word}')
    Y_hat = lm.probabilities(context)
    # # Y_hat = torch.stack(Y_hat)
    print(f'Prob dado contexto:{Y_hat}', Y_hat.dtype)
    Y = torch.Tensor(vec.token_to_index(next_word)).to(torch.int64)
    print(f'Índice de la siguiente palabra: {Y}', Y.dtype)
    perdida = lm.loss_func(Y_hat, Y)
    print(f'Valor función pérdida actual: {perdida}')


def test_training():
    texto = ['Ana Beto Carlos David']
    lm = FFNLM(vectorizer=Vectorizer(texto),
               window_length=2,
               hidden_size=20)
    parameters = {"learning_rate":1e-3,
                "window_length": 2,
                "batch_size":2,
                "num_epochs":500
    }
    lm.train(texto=texto, parametros=parameters)
    ds = LMDataset(texto=texto, window_length=2)
    ds_loader = DataLoader(ds, batch_size=2, shuffle=True)
    for ds_features, ds_labels in ds_loader:
        ds_features = [[x[i] for x in ds_features] for i in range(2)]
        print(f'contexto:{ds_features}; siguiente palabra: {ds_labels}')
        print(lm.probability(ds_labels, ds_features)) 

