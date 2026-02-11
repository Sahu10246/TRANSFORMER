import torch
from transformer import Transformer
from tokenizer import Tokenizer
from config import *
import pandas as pd


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    tokenizer = Tokenizer()
    tokenizer.word2idx = checkpoint["tokenizer"]
    tokenizer.idx2word = {v:k for k,v in tokenizer.word2idx.items()}

    model = Transformer(len(tokenizer.word2idx)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def greedy_decode(model, tokenizer, text, max_len=50):

    tokens = tokenizer.encode(text)
    src = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

    tgt = torch.tensor([[1]]).to(DEVICE)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(src, tgt)
            next_token = output[:, -1].argmax(dim=-1).unsqueeze(0)
            tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == 2:
            break

    return tokenizer.decode(tgt.squeeze().tolist())


def beam_search(model, tokenizer, text, beam_width=3, max_len=50):

    tokens = tokenizer.encode(text)
    src = torch.tensor(tokens).unsqueeze(0).to(DEVICE)

    sequences = [[torch.tensor([1]).to(DEVICE), 0]]

    for _ in range(max_len):
        all_candidates = []

        for seq, score in sequences:
            with torch.no_grad():
                output = model(src, seq.unsqueeze(0))
                probs = torch.log_softmax(output[:, -1], dim=-1)

            topk = torch.topk(probs, beam_width)

            for i in range(beam_width):
                candidate = [
                    torch.cat([seq, topk.indices[0][i].unsqueeze(0)]),
                    score - topk.values[0][i].item()
                ]
                all_candidates.append(candidate)

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    return tokenizer.decode(sequences[0][0].tolist())


if __name__ == "__main__":
    model, tokenizer = load_model()
    text = input("Enter Hindi article: ")
    print("\nGreedy Summary:")
    print(greedy_decode(model, tokenizer, text))
    print("\nBeam Search Summary:")
    print(beam_search(model, tokenizer, text))
