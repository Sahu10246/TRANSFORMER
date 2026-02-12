import re
import collections
import json

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.bpe_codes = {}
        self.word_freq = collections.Counter()
        self.word2idx = {}
        self.idx2word = {}

        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    # ----------------------------
    # 1. Preprocess text for data
    # ----------------------------
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ----------------------------
    # 2. Build initial vocabulary
    # ----------------------------
    def build_vocab(self, texts):

        for text in texts:
            text = self.preprocess(text)
            for word in text.split():
                self.word_freq[" ".join(list(word)) + " </w>"] += 1

        vocab = self.word_freq.copy()

        while len(vocab) < self.vocab_size:

            pairs = self.get_pair_frequencies(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.bpe_codes[best_pair] = len(self.bpe_codes)

        tokens = set()
        for word in vocab:
            tokens.update(word.split())

        tokens = self.special_tokens + list(tokens)

        for idx, token in enumerate(tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    # ----------------------------
    # 3. Get pair frequencies
    # ----------------------------
    def get_pair_frequencies(self, vocab):
        pairs = collections.Counter()

        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq

        return pairs

    # ----------------------------
    # 4. Merge best pair
    # ----------------------------
    def merge_vocab(self, pair, vocab):
        merged_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word in vocab:
            merged_word = pattern.sub("".join(pair), word)
            merged_vocab[merged_word] = vocab[word]

        return merged_vocab

    # ----------------------------
    # 5. Encode
    # ----------------------------
    def encode(self, text):
        text = self.preprocess(text)
        tokens = []

        for word in text.split():
            word = " ".join(list(word)) + " </w>"
            word_tokens = word.split()

            while True:
                pairs = [(word_tokens[i], word_tokens[i+1]) 
                         for i in range(len(word_tokens)-1)]

                merge_candidates = [pair for pair in pairs if pair in self.bpe_codes]

                if not merge_candidates:
                    break

                best_pair = min(merge_candidates, key=lambda p: self.bpe_codes[p])
                i = pairs.index(best_pair)
                word_tokens[i:i+2] = ["".join(best_pair)]

            tokens.extend(word_tokens)

        return [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]

    # ----------------------------
    # 6. Decode
    # ----------------------------
    def decode(self, token_ids):
        tokens = [self.idx2word.get(i, "<unk>") for i in token_ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    # ----------------------------
    # 7. Save / Load
    # ----------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "bpe_codes": self.bpe_codes,
                "word2idx": self.word2idx
            }, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.bpe_codes = {tuple(k.split(",")):v for k,v in data["bpe_codes"].items()}
        self.word2idx = data["word2idx"]
        self.idx2word = {v:k for k,v in self.word2idx.items()}
