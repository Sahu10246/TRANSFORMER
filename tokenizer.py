import re
from collections import Counter

class Tokenizer:
    def __init__(self, max_vocab=20000):
        self.word2idx = {"<pad>":0,"<sos>":1,"<eos>":2,"<unk>":3}
        self.idx2word = {0:"<pad>",1:"<sos>",2:"<eos>",3:"<unk>"}
        self.max_vocab = max_vocab

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(re.findall(r'\w+', text.lower()))
        for i,(w,_) in enumerate(counter.most_common(self.max_vocab),start=4):
            self.word2idx[w]=i
            self.idx2word[i]=w

    def encode(self,text):
        return [self.word2idx.get(w,3) for w in re.findall(r'\w+', text.lower())]

    def decode(self,tokens):
        return " ".join([self.idx2word.get(t,"<unk>") for t in tokens])
