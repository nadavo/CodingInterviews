from typing import List, Any

def binary_search(arr: List[Any], i: Any) -> int:
        l, r = 0, len(arr) - 1
        while l <= r:
            m = (l + r) // 2
            if arr[m] == i:
                return m
            elif arr[m] > i:
                r = m - 1
            else:
                l = m + 1
        return -1

def binary_search_recursive(arr: List[Any], l: int, r: int, val: Any) -> int:
    if l <= r:
        m = (l + r) // 2
        if arr[m] == val:
            return m
        elif arr[m] > val:
            return binary_search_recursive(arr, l, m - 1, val)
        else:
            return binary_search_recursive(arr, m + 1, r, val)
    else:
        return -1


def findMedian(A):
    min_val = int(10e9)
    max_val = 1
    for row in A:
        if max_val < row[-1]:
            max_val = row[-1]
        if min_val > row[0]:
            min_val = row[0]
    median_idx = (len(A) * len(A[0])) // 2
    while min_val <= max_val:
        median_val = (max_val + min_val) // 2
        k = 0
        val_found = False
        for row in A:
            num_smaller, exists = binary_search(row, median_val)
            if num_smaller:
                k += num_smaller
            if exists:
                val_found = True
        if k == median_idx:
            if val_found:
                return median_val
            else:
                
        elif k > median_idx:
            max_val = median_val - 1
        else:
            min_val = median_val + 1


if __name__ == "__main__":
    A = [ 
            [1, 3, 5],
            [2, 6, 9],
            [3, 6, 9]
        ]
    # print(findMedian(A))
    A = [ 
            [1, 1, 3, 3, 3]
        ]
    print(findMedian(A))