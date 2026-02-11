import torch
from torch.utils.data import Dataset
import pandas as pd
from config import MAX_LEN

class HindiDataset(Dataset):
    def __init__(self,csv_path,tokenizer):
        self.data=pd.read_csv(csv_path)
        self.tokenizer=tokenizer

    def pad(self,tokens):
        tokens=tokens[:MAX_LEN]
        return tokens+[0]*(MAX_LEN-len(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        row=self.data.iloc[idx]
        src=self.pad(self.tokenizer.encode(row["article"]))
        tgt=self.pad([1]+self.tokenizer.encode(row["summary"])+[2])
        return torch.tensor(src),torch.tensor(tgt)
