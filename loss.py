import torch.nn as nn
import torch

class LabelSmoothingLoss(nn.Module):
    def __init__(self,smoothing,vocab_size):
        super().__init__()
        self.smoothing=smoothing
        self.vocab=vocab_size

    def forward(self,pred,target):
        confidence=1-self.smoothing
        smooth=self.smoothing/(self.vocab-1)
        one_hot=torch.full_like(pred,smooth)
        one_hot.scatter_(1,target.unsqueeze(1),confidence)
        return torch.mean(torch.sum(-one_hot*torch.log_softmax(pred,dim=1),dim=1))
