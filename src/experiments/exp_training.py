from utils.utils import LMDataset
import torch
from torch.utils.data import DataLoader
from utils.utils import Vectorizer
from lms.models import FFNLM, ZLT
from pathlib import Path
import pandas as pd
import os

data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'clean', 'wiki')


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
    texto = ['ana beto carlos ana beto daniel ana beto edgar ana beto felipe ana beto gabriela']
    window_length = 2
    batch_size = 8
    lm = ZLT(vectorizer=Vectorizer(texto),
               window_length=window_length)
    parameters = {"learning_rate":1e-2,
                "window_length":window_length,
                "batch_size":batch_size,
                "num_epochs":1000
    }
    print(lm.FFN.device)
    lm.train(texto=texto, parametros=parameters)
    ds = LMDataset(texto=texto, window_length=window_length, vectorizer=lm.vectorizer)
    ds_loader = DataLoader(ds, batch_size=1, shuffle=False)
    probas = []
    df_features = []
    df_labels = []
    for ds_features, ds_labels in ds_loader:
        batch_len = len(ds_features[0])
        ds_features = [[x[i] for x in ds_features] for i in range(batch_len)]
        ds_labels = list(ds_labels)
        #print(f'contexto:{ds_features}; siguiente palabra: {ds_labels}')
        #print(lm.probability(ds_labels, ds_features)) 
        probas.append(lm.probability(ds_labels, ds_features))
        df_features.append(ds_features[0])
        df_labels.append(ds_labels[0])

    data = {'Contexto': df_features, 'SgtePalabra': df_labels, 'Probabilidad': probas}
    probabilities_df = pd.DataFrame(data)
    print(probabilities_df)
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
    window_length = 3
    batch_size = 16
    
    # ---------------------------------------
    # Descomentar cada cambio de parámetros
    vec = Vectorizer(texto)
    vec.save_vocabulary()
    # ---------------------------------------
    vec = Vectorizer(None)

    lm = FFNLM(vectorizer=vec,
               window_length=window_length,
               hidden_size=20)
    print('vocabulary_size: ',lm.vocabulary_size)
    #print('vocabulary: ',lm.vectorizer.tokens)
    # --------------------------------------
    # Training
    # --------------------------------------
    parameters = {"learning_rate":1e-4,
                "window_length":window_length,
                "batch_size":batch_size,
                "num_epochs":5
    }
    print(lm.FFN.device)
    lm.summary()
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
