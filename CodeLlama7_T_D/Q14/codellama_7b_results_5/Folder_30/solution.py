from typing import List


def find_second_smallest_num(l: List[int]) -> int | None:
    l = l[26:52 + 1]
    if len(l) < 2:
        return None
    l.sort()
    return l[1]