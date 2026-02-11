import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from config import *
from tokenizer import BPETokenizer
from dataset import HindiDataset
from transformer import Transformer
from loss import LabelSmoothingLoss
from metrics import rouge_scores
import pandas as pd
import os

def transformer_lr(step, d_model, warmup):
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def train():

    print("Loading dataset...")
    df = pd.read_csv("hindi_dataset.csv")

    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.build_vocab(df["article"])
    tokenizer.save("bpe_tokenizer.json")    

    dataset = HindiDataset("hindi_dataset.csv", tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Transformer(len(tokenizer.word2idx)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = LabelSmoothingLoss(LABEL_SMOOTHING, len(tokenizer.word2idx))

    scaler = GradScaler()
    best_loss = float("inf")

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            global_step += 1
            lr = transformer_lr(global_step, EMBED_DIM, WARMUP_STEPS)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()

            with autocast():
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer.word2idx
            }, MODEL_PATH)
            print("Best model saved.")

    print("Training complete.")


if __name__ == "__main__":
    train()
