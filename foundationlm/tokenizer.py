from pathlib import Path
import json


class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.inverse_vocab = {value: key for key, value in vocab.items()}

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], data["merges"])

    def encode(self, text):
        tokens = [str(byte) for byte in text.encode("utf-8")]
        merge_ranks = {tuple(pair): rank for rank, pair in enumerate(self.merges)}

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            ranked = [(merge_ranks[p], i, p) for i, p in enumerate(pairs) if p in merge_ranks]

            if not ranked:
                break

            _, index, pair = min(ranked)
            merged = pair[0] + "_" + pair[1]
            tokens = tokens[:index] + [merged] + tokens[index + 2:]

        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids):
        parts = [self.inverse_vocab[token_id] for token_id in token_ids]
        byte_values = []

        for part in parts:
            for item in part.split("_"):
                byte_values.append(int(item))

        return bytes(byte_values).decode("utf-8", errors="replace")
