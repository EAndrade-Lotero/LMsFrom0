import copy
import torch
from torch import nn
from typing import Optional


from lms.transformers import (
	EncoderDecoder,
	MultiHeadedAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
    Embeddings,
    Generator
)

def make_model(
    src_vocab:int, 
    tgt_vocab:int, 
    N:Optional[int]=6, 
    d_model:Optional[int]=512, 
    d_ff:Optional[int]=2048, 
    h:Optional[int]=8, 
    dropout:Optional[float]=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

test_model = make_model(
    src_vocab=5, 
    tgt_vocab=5, 
    N=1, 
    d_model=4, 
    d_ff=8, 
    h=2, 
    dropout=0
)

src = torch.LongTensor([[1, 2, 3]])
print('src.shape', src.shape)
print(src)

src_mask = torch.ones(1, 1, 3)
print('src_mask.shape', src_mask.shape)
print(src_mask)

memory = test_model.encode(src, src_mask)
print('src.shape', src.shape)
print(memory)