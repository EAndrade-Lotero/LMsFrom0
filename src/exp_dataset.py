from utils import LMDataset
import torch
from torch.utils.data import DataLoader

def test_dataset():

    texto = ['Qué linda que está la luna, colgada como una fruta.', 'Si se llegara a caer, que golpe tan tenaz.']
    window_length = 2
    batch_size = 3
    ds = LMDataset(texto=texto, window_length=window_length)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ds_features, ds_labels in ds_loader:
        ds_features = [[x[i] for x in ds_features] for i in range(batch_size)]
        print(ds_features, ds_labels)