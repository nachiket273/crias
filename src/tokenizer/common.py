from collections import Counter
from itertools import islice, tee
from typing import Int, List

__all__ = ['get_stats', 'merge']


def get_stats(ids : List[Int]) -> Counter:
    r"""
    Given a list of ids, return the count of consecutive id pairs.

    Args:
        ids(list of int) List of ids

    Example:
        >>> ids = [1, 2, 4, 1, 3, 1, 2, 4]
        >>> cnt = get_stats(ids)
        Counter({(1, 2): 2, (2, 4): 2, (4, 1): 1, (1, 3): 1, (3, 1): 1})
    """
    iters = tee(ids, 2)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return Counter(zip(*iters))

def merge(ids, pair, idx):
    r"""
    
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
