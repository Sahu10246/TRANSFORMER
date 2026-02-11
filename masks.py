import torch

def padding_mask(seq,pad_idx=0):
    return (seq!=pad_idx).unsqueeze(1).unsqueeze(2)

def causal_mask(size):
    mask=torch.tril(torch.ones(size,size))
    return mask.unsqueeze(0).unsqueeze(0)
