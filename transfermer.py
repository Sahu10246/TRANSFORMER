import torch.nn as nn
from layers import *
from masks import *

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=MultiHeadAttention()
        self.norm1=LayerNorm(EMBED_DIM)
        self.ff=FeedForward()
        self.norm2=LayerNorm(EMBED_DIM)

    def forward(self,x,mask):
        x=self.norm1(x+self.attn(x,x,x,mask))
        x=self.norm2(x+self.ff(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=MultiHeadAttention()
        self.norm1=LayerNorm(EMBED_DIM)
        self.cross_attn=MultiHeadAttention()
        self.norm2=LayerNorm(EMBED_DIM)
        self.ff=FeedForward()
        self.norm3=LayerNorm(EMBED_DIM)

    def forward(self,x,enc_out,src_mask,tgt_mask):
        x=self.norm1(x+self.self_attn(x,x,x,tgt_mask))
        x=self.norm2(x+self.cross_attn(x,enc_out,enc_out,src_mask))
        x=self.norm3(x+self.ff(x))
        return x


class Transformer(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,EMBED_DIM)
        self.encoder=nn.ModuleList([EncoderBlock() for _ in range(NUM_LAYERS)])
        self.decoder=nn.ModuleList([DecoderBlock() for _ in range(NUM_LAYERS)])
        self.fc=nn.Linear(EMBED_DIM,vocab_size)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self,src,tgt):
        src_mask=padding_mask(src)
        tgt_mask=padding_mask(tgt)&causal_mask(tgt.size(1)).to(tgt.device)

        src=self.embed(src)
        tgt=self.embed(tgt)

        for layer in self.encoder:
            src=layer(src,src_mask)

        for layer in self.decoder:
            tgt=layer(tgt,src,src_mask,tgt_mask)

        return self.fc(tgt)
