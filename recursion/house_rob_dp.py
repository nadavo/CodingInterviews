from typing import List, Dict, Callable
from timeit import default_timer as timer


def rob_recursion(nums: List[int], i: int):
    if i < 0:
        return 0
    return max(rob_recursion(nums, i - 2) + nums[i], rob_recursion(nums, i - 1))


def rob_recursion_memo(nums: List[int], i: int, memo: Dict[int, int]):
    if i < 0:
        return 0
    elif i in memo:
        return memo[i]
    else:
        result = max(
            rob_recursion_memo(nums, i - 2, memo) + nums[i],
            rob_recursion_memo(nums, i - 1, memo),
        )
        memo[i] = result
        return result


def rob_iterative_memo(nums: List[int]):
    memo = [0, nums[0]]
    for i in range(1, len(nums), 1):
        memo.append(max(memo[i], memo[i - 1] + nums[i]))
    return memo[-1]


def rob_iterative_vars(nums: List[int]):
    max_val, last_max_val = 0, 0
    for num in nums:
        temp = max_val
        max_val = max(max_val, last_max_val + num)
        last_max_val = temp
    return max_val


def rob(nums: List[int], rob_fn: Callable, rob_fn_args: List) -> int:
    if not nums:
        return 0
    elif len(nums) == 1:
        return nums[0]
    elif len(nums) == 2:
        return max(nums)
    elif len(nums) <= 4:
        odd_total, even_total, edge_total = (
            nums[1],
            nums[0] + nums[2],
            nums[0] + nums[-1],
        )
        if len(nums) == 4:
            odd_total += nums[3]
        return max(odd_total, even_total, edge_total)
    else:
        return rob_fn(nums, *rob_fn_args)


def test(nums: List[int], rob_fn: Callable, rob_fn_args: List, expected: int):
    """Run test on given list, rob method and validate output"""
    start = timer()
    output = rob(nums, rob_fn, rob_fn_args)
    end = timer()
    print(
        f"{rob_fn.__name__.title()}: {output == expected}, {output}, {end - start:.8f}"
    )


def main():
    tests = [
        ([1, 2, 3, 1], 4),
        ([1, 4, 3, 1], 5),
        ([2, 1, 1, 2], 4),
        ([2, 7, 9, 3, 1, 4, 5, 6, 8, 3], 25),
    ]
    for nums, expected in tests:
        print(nums)
        test(nums, rob_recursion, [len(nums) - 1], expected)
        test(nums, rob_recursion_memo, [len(nums) - 1, {}], expected)
        test(nums, rob_iterative_memo, [], expected)
        test(nums, rob_iterative_vars, [], expected)


if __name__ == "__main__":
    main()
