from typing import List, Set
from math import sqrt


def _repeated_val(digits: Set[str], val: str) -> bool:
    if val and val != ".":
        if val in digits:
            return True
        else:
            digits.add(val)
    return False


def _validate_rows(board: List[List[str]]) -> bool:
    for row in board:
        r_digits = set()
        for i in filter(lambda x: x and x != ".", row):
            if _repeated_val(r_digits, i):
                return False
    return True


def _validate_columns(board: List[List[str]]) -> bool:
    for c in range(len(board[0])):
        c_digits = set()
        for row in board:
            if _repeated_val(c_digits, row[c]):
                return False
    return True


def _validate_block(
    board: List[List[str]], r_range: List[int], c_range: List[int]
) -> bool:
    b_digits = set()
    for i in r_range:
        for j in c_range:
            if _repeated_val(b_digits, board[i][j]):
                return False
    return True


def isValidSudoku(board: List[List[str]]) -> bool:
    if not _validate_rows(board):
        return False

    if not _validate_columns(board):
        return False

    s = int(sqrt(len(board)))
    blocks_ranges = list(list(range(k, k + s)) for k in range(0, len(board), s))
    for r_range in blocks_ranges:
        for c_range in blocks_ranges:
            if not _validate_block(board, r_range, c_range):
                return False

    return True


def isValidSudoku_space(board: List[List[str]]) -> bool:
    n = len(board)
    k = int(sqrt(n))

    rows = [set() for _ in range(n)]
    columns = [set() for _ in range(n)]
    blocks = [set() for _ in range(n)]

    for i in range(n):
        for j in range(n):
            val = board[i][j]

            if not val or val == ".":
                continue

            if val in rows[i]:
                return False
            rows[i].add(val)

            if val in columns[j]:
                return False
            columns[j].add(val)

            block_idx = (i // k) * k + (j // k)
            if val in blocks[block_idx]:
                return False
            blocks[block_idx].add(val)

    return True
