from collections import Counter
from itertools import islice, tee
from typing import Int, List, Tuple

__all__ = ['get_stats', 'merge', 'Tokenizer']


def get_stats(ids : List[Int]) -> Counter:
    r"""
    Given a list of ids, return the count of consecutive id pairs.

    Args:
        ids(list of int) List of ids

    Return:
        Counter of occurances of pairs

    Example:
        >>> ids = [1, 2, 4, 1, 3, 1, 2, 4]
        >>> cnt = get_stats(ids)
        Counter({(1, 2): 2, (2, 4): 2, (4, 1): 1, (1, 3): 1, (3, 1): 1})
    """
    iters = tee(ids, 2)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return Counter(zip(*iters))

def merge(ids: List[Int], pair: Tuple[Int, Int], idx: int) -> List[Int]:
    r"""
    Replace each occurance of pair in list of ids with new given id.

    Args:
        ids(list of int)   List of ids
        pair(tuple of int) Tuple of ids- pair to be replaced.
        idx(int)           Id with which the pair will be replaced.

    Return:
        new_ids: List of ids

    Example:
        >>> ids = [1, 2, 4, 1, 3, 1, 2, 4]
        >>> pair = (1, 2)
        >>> new_ids = merge(ids, pair, 5)
        >>> new_ids
        [5, 4, 1, 3, 5, 4]    
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


######################################################################
#                                                                    #
#                        Base Tokenizer Class                        #
#                                                                    #
######################################################################


class Tokenizer:
    r"""
    Base Class for tokenizers.

    Specific tokenizer will implement its own train, enocde and decode methods
    save and load will be common for all tokenizers.
    """
    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = {idx:bytes([idx]) for idx in range(256)}

    def train(self, text: str):
        raise NotImplementedError
    
    def encode(self, text: str):
        raise NotImplementedError
    
    def decode(self, ids : List[Int]):
        raise NotImplementedError
    
    def build_vocab(self):
        for (p1, p2), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p1] + self.vocab[p2]
        for tok, idx in self.special_tokens.items():
            self.vocab[idx] = tok.encode("utf-8")

    def save_model(self, path: str):
        # TO-DO: Implement
        pass
    
    def load_model(self, path: str):
        # TO-DO: Implement
        pass
