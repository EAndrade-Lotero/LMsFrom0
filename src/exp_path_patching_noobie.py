from utils import LMDataset
import torch
from torch.utils.data import DataLoader
from utils import Vectorizer
from models import FFNLM

def test_path():
    texto = ['Ana Beto Carlos David']
    lm = FFNLM(vectorizer=Vectorizer(texto),
               window_length=2,
               hidden_size=20)
    lm.load_model()
    layers = ['fc1.weight', 'fc_intermediate.weight', 'fc2.weight']
    dict_layers = lm.FFN.named_parameters()
    batch_size = 1
    ds = LMDataset(texto=texto, window_length=2)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for ds_features, ds_labels in ds_loader:
        ds_features = [[x[i] for x in ds_features] for i in range(batch_size)]
        print(f'contexto:{ds_features}; siguiente palabra: {ds_labels}')
        probs = lm.probabilities(ds_features)
        print(probs) 
        maximo = torch.argmax(probs)
        print(maximo)
        layer = layers[2]
        weights = lm.FFN.fc2.weight
#        print(weights)
        print(weights.shape)
        weights_best_neuron = weights[maximo]
        print(weights_best_neuron.shape)
        print(weights_best_neuron)
