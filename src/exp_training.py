from utils import LMDataset
import torch
from torch.utils.data import DataLoader
from utils import Vectorizer
from models import FFNLM
from pathlib import Path
import os

data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'clean')


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


def test_training_simple():
    texto = ['Ana Beto Carlos Ana Beto David']
    window_length = 3
    batch_size = 2
    lm = FFNLM(vectorizer=Vectorizer(texto),
               window_length=window_length,
               hidden_size=20)
    parameters = {"learning_rate":1e-2,
                "window_length":window_length,
                "batch_size":batch_size,
                "num_epochs":500
    }
    lm.train(texto=texto, parametros=parameters)
    ds = LMDataset(texto=texto, window_length=window_length)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ds_features, ds_labels in ds_loader:
        batch_len = len(ds_features[0])
        ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
        ds_labels = list(ds_labels)
        print(f'contexto:{ds_features}; siguiente palabra: {ds_labels}')
        print(lm.probability(ds_labels, ds_features)) 
    lm.save_model() 


def test_training():
    # --------------------------------------
    # Loading corpus
    # --------------------------------------
    print('Loading corpus...')
    texto = ''
    lista_textos = [f for f in os.listdir(data_folder) if f.split('.')[-1] == 'txt']
    for wiki_txt in lista_textos:
        print(f'\tReading {wiki_txt}...')
        with open(data_folder / Path(wiki_txt), encoding='utf-8') as fp:
            texto += fp.read()
        fp.close()
        print(f'¡Ok! Texto de longitud {len(texto.split())}')
    # --------------------------------------
    # Loading Language Model
    # --------------------------------------
    window_length = 2
    batch_size = 16
    lm = FFNLM(vectorizer=Vectorizer(texto),
               window_length=window_length,
               hidden_size=20)
    # --------------------------------------
    # Training
    # --------------------------------------
    parameters = {"learning_rate":1e-4,
                "window_length":window_length,
                "batch_size":batch_size,
                "num_epochs":50
    }
    print('Training...')
    lm.train(texto=texto, parametros=parameters)
    lm.save_model() 
    # --------------------------------------
    # Finding perplexity
    # --------------------------------------
    print('Text perplexity:', lm.perplexity(texto))


def test_corpus():
    # --------------------------------------
    # Loading corpus
    # --------------------------------------
    print('Loading corpus...')
    texto = ''
    lista_textos = [f for f in os.listdir(data_folder) if f.split('.')[-1] == 'txt']
    for wiki_txt in lista_textos:
        print(f'\tReading {wiki_txt}...')
        with open(data_folder / Path(wiki_txt), encoding='utf-8') as fp:
            texto += fp.read()
        fp.close()
        print(f'¡Ok! Texto de longitud {len(texto.split())}')
    window_length = 2
    batch_size = 16
    ds = LMDataset(texto=texto, window_length=window_length)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for ds_features, ds_labels in ds_loader:
        # Reconfiguramos los features
        batch_len = len(ds_features[0])
        ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
        # Reconfiguramos los targets
        ds_labels = list(ds_labels)
        print(ds_features, ds_labels)
