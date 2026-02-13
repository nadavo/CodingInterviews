from typing import List, Tuple

"""Remove all occurrences of 'a' and duplicate all occurrences of 'b' from a given string"""


def solution(A: List[str]) -> int:
    print("Input:", A)
    l_idx, r_idx = 0, 0
    num_garbage = calc_num_garbage(A)
    l_idx, r_idx = remove_a(A, l_idx, r_idx)
    l_idx -= 1
    r_idx = len(A) - 1 - num_garbage
    remove_b(A, l_idx, r_idx)
    if num_garbage > 0:
        A = A[:-num_garbage]
    print("Output:", A)
    return len(A)


def calc_num_garbage(A: List[str]) -> int:
    num_garbage = 0
    for i in A:
        if i == "a":
            num_garbage += 1
        elif i == "b":
            num_garbage -= 1
    return num_garbage


def remove_a(A: List[str], l_idx: int, r_idx: int) -> Tuple[int, int]:
    while l_idx <= r_idx and r_idx < len(A):
        while r_idx < len(A) and A[r_idx] == "a":
            r_idx += 1
        A[l_idx] = A[r_idx]
        l_idx += 1
        r_idx += 1
    return l_idx, r_idx


def remove_b(A: List[str], l_idx: int, r_idx: int):
    while l_idx >= 0 and r_idx >= l_idx:
        A[r_idx] = A[l_idx]
        r_idx -= 1
        if A[l_idx] == "b":
            A[r_idx] = A[l_idx]
            r_idx -= 1
        l_idx -= 1


if __name__ == "__main__":
    solution(list("ccd"))
    solution(list("abcd"))
    solution(list("aababccd"))

# Reader-Writer Pattern (more practice)
# Pseudo-code before translating from board to code
# If stuck on code go back to board in order to regain focus
# Draw the example patterns on the board - use board as much possible
# Answer with confidence when I am 95% right don't underestimate knowledge
# Run simple test on board after code is written
# Understand time complexity of basic python operations
# Comments in code for corner cases - don't linger
# Always start and build up from the naive solution towards a better/best solution, and verbalize my thought process to leave impression
# Unblock myself by going back to the board and run the simplest solution
# Practice translation from board to code - first write down full solution on example on paper, then translate to code
# Interviewer is trying to asses: algorithmic thinking, work chemistry, communication skills, coding skills.
