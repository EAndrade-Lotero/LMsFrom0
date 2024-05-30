import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroLayerTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super(ZeroLayerTransformer, self).__init__()

        # Atención sin capas ocultas
        self.attention = nn.MultiheadAttention(d_model, nhead)

        # Capa de retroalimentación (feedforward)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Normalización y dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Atención sin capas ocultas
        attn_output, _ = self.attention(x, x, x)

        # Agregar retroalimentación y aplicar normalización y dropout
        x = x + self.dropout(attn_output)
        x = self.norm(x)

        # Feedforward
        ff_output = self.feedforward(x)

        # Agregar retroalimentación y aplicar normalización y dropout
        x = x + self.dropout(ff_output)
        x = self.norm(x)

        return x

# Ejemplo de uso
input_tensor = torch.rand((10, 32, 512))  # Secuencia de longitud 10, lote de tamaño 32, embedding dimension de 512
zero_layer_transformer = ZeroLayerTransformer()
output = zero_layer_transformer(input_tensor)
print(output.shape)
