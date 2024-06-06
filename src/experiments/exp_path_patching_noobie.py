from utils.utils import LMDataset
import torch
from torch.utils.data import DataLoader
from utils.utils import Vectorizer
from lms.models import FFNLM

def test_perplexity():
    texto = ['Ana Beto Carlos Ana Beto David']
    lm = FFNLM(vectorizer=Vectorizer(texto),
                window_length=3,
                hidden_size=20)
    lm.load_model()
    perplexity = lm.perplexity(texto)
    print(f'perplexity: {perplexity}')

def test_path():
    texto = ['Ana Beto Carlos Ana Beto David']
    batch_size = 1
    ds = LMDataset(texto=texto, window_length=3)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    lm = FFNLM(vectorizer=Vectorizer(texto),
                window_length=3,
                hidden_size=20)
    lm.load_model()
    for ds_features, ds_labels in ds_loader:
        print('')
        print('-'*30)
        ds_features = [[x[i] for x in ds_features] for i in range(batch_size)]
        print(f'contexto:{ds_features}; siguiente palabra: {ds_labels}')
        coded_context = lm.code_context(ds_features).to(torch.float32)
        activaciones_1_layer = lm.FFN.fc1(coded_context)
        activaciones_1_layer = lm.FFN.activation_fc1(activaciones_1_layer)
        probs_1_layer = lm.FFN.fc2(activaciones_1_layer)
        probs_1_layer = lm.FFN.activation_fc2(probs_1_layer, dim=1)
        print('Predicción primera capa:')
        print(probs_1_layer) 
        maximo = torch.argmax(probs_1_layer)
        print(maximo, lm.vectorizer.index_to_token([maximo]))
        print('')
        activaciones_2_layer = lm.FFN.fc_intermediate_1(activaciones_1_layer)
        activaciones_2_layer = lm.FFN.activation_fci(activaciones_2_layer, dim=1)
        probs_2_layer = lm.FFN.fc2(activaciones_2_layer)
        probs_2_layer = lm.FFN.activation_fc2(probs_2_layer, dim=1)
        print('Predicción segunda capa:')
        print(probs_2_layer) 
        maximo = torch.argmax(probs_2_layer)
        print(maximo, lm.vectorizer.index_to_token([maximo]))
        print('')
        print('Predicción toda la red (3a capa)')
        probs_last = lm.probabilities(ds_features)
        print(probs_last) 
        maximo = torch.argmax(probs_last)
        print(maximo, lm.vectorizer.index_to_token([maximo]))


