from collections import Counter
from itertools import islice, tee
from typing import Int, List


def get_stats(ids : List[Int]) -> Counter:
    """
    Given a list of ids, return the count of consecutive id pairs.
    """
    iters = tee(ids, 2)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return Counter(zip(*iters))

def merge(ids, pair, idx):
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
