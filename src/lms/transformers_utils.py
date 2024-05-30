import copy
import math
import torch
from torch import nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    # print('--- Attention layer ---')
    # print('\tquery:')
    # print(query)
    # print('\tkey:')
    # print(key)
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # print('\tkey transposed:')
    # print(key.transpose(-2, -1))    
    scores = torch.matmul(query, key.transpose(-2, -1)) 
    # print('\tdot product:')
    # print(scores)        
    scores /= math.sqrt(d_k)
    # print('\tnormalized:')
    # print(scores)        
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # print('\tmasked:')
    # print(scores)        
    p_attn = scores.softmax(dim=-1)
    # print('\tsoftmax attention:')
    # print(p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print('\tvalue:')
    # print(value)
    x = torch.matmul(p_attn, value)
    return torch.matmul(p_attn, value), p_attn