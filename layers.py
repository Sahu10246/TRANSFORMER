import torch
import torch.nn as nn
import math
from config import *

class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(dim))
        self.beta=nn.Parameter(torch.zeros(dim))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps)+self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim=EMBED_DIM//NUM_HEADS

        self.W_q=nn.Linear(EMBED_DIM,EMBED_DIM)
        self.W_k=nn.Linear(EMBED_DIM,EMBED_DIM)
        self.W_v=nn.Linear(EMBED_DIM,EMBED_DIM)
        self.fc=nn.Linear(EMBED_DIM,EMBED_DIM)
        self.dropout=nn.Dropout(DROPOUT)

    def split(self,x):
        b,s,d=x.size()
        x=x.view(b,s,NUM_HEADS,self.head_dim)
        return x.transpose(1,2)

    def combine(self,x):
        b,h,s,d=x.size()
        return x.transpose(1,2).contiguous().view(b,s,EMBED_DIM)

    def forward(self,q,k,v,mask=None):
        Q=self.split(self.W_q(q))
        K=self.split(self.W_k(k))
        V=self.split(self.W_v(v))

        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)

        attn=self.dropout(torch.softmax(scores,dim=-1))
        out=torch.matmul(attn,V)
        out=self.combine(out)
        return self.fc(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(EMBED_DIM,FF_DIM)
        self.fc2=nn.Linear(FF_DIM,EMBED_DIM)
        self.dropout=nn.Dropout(DROPOUT)

    def forward(self,x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))
