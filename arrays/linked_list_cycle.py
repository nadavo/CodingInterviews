from timeit import default_timer as timer
from typing import Any, List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x: Any):
        self.val = x
        self.next = None


visited = set()


def hasCycle_recursive(head: ListNode) -> bool:
    if not head or not head.next:
        return False
    curr = head.next
    if curr in visited:
        return True
    else:
        visited.add(curr)
        return hasCycle_recursive(curr)


def hasCycle_iterative(head: ListNode) -> bool:
    if not head or not head.next:
        return False

    visited = {head}
    exp = head.next
    while exp:
        if exp in visited:
            return True
        else:
            visited.add(exp)
            exp = exp.next

    return False


def hasCycle_constant_memory(head: ListNode) -> bool:
    if not head or not head.next:
        return False
    if head == head.next:
        return True

    s, d = head, head
    while d and d.next:
        s = s.next
        d = d.next.next
        if s == d:
            return True
    return False


def test(l: List[Any], pos: int, expected: bool):
    """Run test on given string and validate output"""
    l_nodes = [ListNode(i) for i in l]
    if l_nodes:
        head = l_nodes[0]
        if pos == -1 or pos >= len(l_nodes):
            l_nodes[-1].next = None
        else:
            l_nodes[-1].next = l_nodes[pos]
        for i in range(len(l_nodes) - 1):
            l_nodes[i].next = l_nodes[i + 1]
    else:
        head = ListNode(None)
    for f in (hasCycle_recursive, hasCycle_iterative, hasCycle_constant_memory):
        start = timer()
        output = f(head)
        end = timer()
        print(
            f"{f.__name__}: {l} -> {output} - {'Correct' if output == expected else 'FAIL'} ({end - start:.8f})"
        )


def main():
    test([3, 2, 0, -4], 1, True)
    test([1, 2], 0, True)
    test([1], -1, False)
    test([], -1, False)


if __name__ == "__main__":
    main()
