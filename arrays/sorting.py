from typing import List, Callable
from timeit import default_timer as timer


class ListNode:
    """Definition for singly-linked list."""

    def __init__(self, x, next=None):
        self.value = x
        self.next = next


class InsertionSort:
    """T=O(n^2), S=O(1)"""

    @staticmethod
    def sort_array_list(arr: List) -> List:
        for i in range(1, len(arr)):
            curr = arr[i]
            # shift all elements greater than curr forward
            j = i - 1
            while j >= 0 and arr[j] > curr:
                arr[j + 1] = arr[j]
                j -= 1
            # place curr in sorted position
            arr[j + 1] = curr
        return arr

    @staticmethod
    def sort_linked_list(node: ListNode) -> List:
        # hold pointer to head of list
        head = ListNode(float("-inf"), node)
        # pointers for list traversal
        prev, curr = node, node
        # traverse till end of list
        while curr:
            # continue if sorted
            if prev <= curr:
                prev = curr
                curr = curr.next
            # sort
            else:
                # init pointers before and after to find correct position of curr
                before = head
                after = head.next
                # connect prev to next node to maintain list after detaching curr
                prev.next = curr.next
                # find correct position for curr (before, curr, after) by incrementing before and after
                while after and curr.value > after.value:
                    before = after
                    after = after.next
                # move curr to its sorted position in list
                before.next = curr
                curr.next = after
                # increment curr to next node
                curr = curr.next
        # values = list()
        # curr = node
        # while curr:
        #     values.append(curr.value)
        #     curr = curr.next
        return head.next


class MergeSort:
    """T=O(nlogn), S=O(n)"""

    @staticmethod
    def divide(arr: List, start: int, end: int) -> None:
        """Divide - split to subarrays"""
        if start < end:
            mid = (start + end) // 2
            MergeSort.divide(arr, start, mid)
            MergeSort.divide(arr, mid + 1, end)
            MergeSort.conquer(arr, start, mid, end)

    @staticmethod
    def conquer(arr: List, start: int, mid: int, end: int) -> None:
        """Conquer - merge sorted subarrays"""
        temp = list()
        l, r = start, mid + 1
        while l <= mid and r <= end:
            if arr[l] <= arr[r]:
                temp.append(arr[l])
                l += 1
            else:
                temp.append(arr[r])
                r += 1
        while l <= mid:
            temp.append(arr[l])
            l += 1
        while r <= end:
            temp.append(arr[r])
            r += 1
        for i, val in enumerate(temp):
            arr[i + start] = val

    @staticmethod
    def sort_array_list(arr: List) -> List:
        MergeSort.divide(arr, 0, len(arr) - 1)
        return arr


class QuickSort:
    """T=O(nlogn), S=O(1)"""

    @staticmethod
    def partition(arr: List, low: int, high: int) -> int:
        # select pivot to be last element in arr
        pivot = arr[high]
        # select element smaller than pivot
        i = low - 1
        for j in range(low, high):
            # swap elements positions if current element is smaller than selected pivot
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    @staticmethod
    def sort(arr: List, low: int, high: int) -> None:
        if low < high:
            part = QuickSort.partition(arr, low, high)
            QuickSort.sort(arr, low, part - 1)
            QuickSort.sort(arr, part + 1, high)

    @staticmethod
    def sort_array_list(arr: List) -> List:
        QuickSort.sort(arr, 0, len(arr) - 1)
        return arr


def test(arr: List, sort_fn: Callable, expected: List):
    start = timer()
    output = sort_fn(arr)
    end = timer()
    print(f"{sort_fn.__name__.title()}: {output == expected}, {output}, {end - start:.8f}")


def main():
    arr = [12, 11, 13, 5, 6, 24, 32, 20, 16]
    expected = sorted(arr)
    test([12, 11, 13, 5, 6, 24, 32, 20, 16], InsertionSort.sort_array_list, expected)
    test([12, 11, 13, 5, 6, 24, 32, 20, 16], MergeSort.sort_array_list, expected)
    test([12, 11, 13, 5, 6, 24, 32, 20, 16], QuickSort.sort_array_list, expected)


if __name__ == "__main__":
    main()
