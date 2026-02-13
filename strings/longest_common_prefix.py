from typing import List


def longestCommonPrefix(strs: List[str]) -> str:
    if not strs:
        return ""
    elif len(strs) == 1:
        return strs[0]

    shortest_s = strs[0]
    min_len = len(shortest_s)
    for s, l in zip(strs[1:], map(len, strs[1:])):
        if l < min_len:
            min_len = l
            shortest_s = s

    if not shortest_s:
        return ""

    for k in range(min_len):
        for i in range(len(strs) - 1):
            if strs[i][k] != strs[i + 1][k]:
                return shortest_s[:k]

    return shortest_s


def longestCommonSuffix(strs: List[str]) -> str:
    if not strs:
        return ""
    elif len(strs) == 1:
        return strs[0]

    shortest_s = strs[0]
    min_len = len(shortest_s)
    for s, l in zip(strs[1:], map(len, strs[1:])):
        if l < min_len:
            min_len = l
            shortest_s = s

    if not shortest_s:
        return ""

    for k in range(1, min_len + 1):
        for i in range(len(strs) - 1):
            if strs[i][-k] != strs[i + 1][-k]:
                return shortest_s[-(k - 1) :] if k > 1 else ""

    return shortest_s
