from typing import List, Callable, Tuple
from timeit import default_timer as timer

def permuteUnique(nums: List[int]) -> List[List[int]]:
    if len(nums) == 1:
        return [[nums[0]]]
    result = list()
    for i in set(nums):
        remaining = list(nums)
        remaining.remove(i)
        for p in permuteUnique(remaining):
            result.append([i] + p)
    return result

def permutations(nums: List[int]) -> List[List[int]]:
    if len(nums) == 1:
        return [[nums[0]]]
    result = list()
    for i in range(len(nums)):
        for p in permutations(nums[1:]):
            p.insert(i, nums[0])
            result.append(p)
    return result


def test(test_case: Tuple, fn: Callable):
    arr, expected = test_case
    start = timer()
    output = fn(arr)
    end = timer()
    print(f"{fn.__name__.title()}: {output == expected}, {output}, {end - start:.8f}")


if __name__ == "__main__":
    test1 = ([1,2,3], [[1,2,3],[1,3,2],[2,1,3],[3,1,2],[2,3,1],[3,2,1]])
    test2 = ([1,1,2], [[1,1,2],[1,2,1],[2,1,1]])
    test(test1, permutations)
    test(test2, permuteUnique)
