# Very basic, minial implementation of byte-pair encoding tokenizer.
# Follows: https://en.wikipedia.org/wiki/Byte_pair_encoding
# Without any regex handling and/or any special token handling
from common import get_stats, merge
from common import Tokenizer

from typing import Int, List

class BasicBPE(Tokenizer):
    def __init__(self, vocab_size: int=32000) -> None:
        super().__init__()
        assert (vocab_size > 256, f"Vocabulary size should be more than 256.")
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256

    def train(self, text: str) -> None:
        text_bytes = bytes(text)
        ids = list(text_bytes)
        for i in range(self.num_merges):
            stats = get_stats(ids)
            pair = stats.most_common(1)[0][0]
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx

        self.build_vocab()

    def encode(self, text: str) -> List[Int]:
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while(len(ids) >= 2):
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
    
    def decode(self, ids: List[Int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
