from timeit import default_timer as timer
from typing import Callable


# String to Signed Integer
# https://leetcode.com/explore/interview/card/top-interview-questions-easy/127/strings/884/
def myAtoi(s: str) -> int:
    s = s.lstrip(" ")
    if not s:
        return 0

    if len(s) == 1:
        return int(s) if s[0].isdigit() else 0

    if s[0] not in ("-", "+") and not s[0].isdigit():
        return 0

    l = 0
    sign = 1
    max_i = 2**31 - 1
    if s[0] == "-":
        l = 1
        sign = -1
        max_i += 1
    elif s[0] == "+":
        l = 1

    s = s.lstrip("0")
    if not s or not s[l].isdigit():
        return 0

    r = l
    while r < len(s) and s[r].isdigit():
        r += 1

    return sign * min(int(s[l:r]), max_i)


def test(s: str, expected: int, f: Callable = myAtoi):
    """Run test on given string and validate output"""
    start = timer()
    output = f(s)
    end = timer()
    print(
        f"{f.__name__}: {s} -> {output} - {'Correct' if output == expected else 'FAIL'} ({end - start:.8f})"
    )


def main():
    test("42", 42)
    test(" -042", -42)
    test("1337c0d3", 1337)
    test("0-1", 0)
    test("words and 987", 0)


if __name__ == "__main__":
    main()
