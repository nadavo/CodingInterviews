from typing import List, Tuple


def maxSubArray(nums: List[int]) -> int:
    if not nums:
        return 0
    elif len(nums) == 1:
        return nums[0]
    elif len(nums) == 2:
        return max(*nums, sum(nums))

    max_num = max(nums)
    if max_num <= 0 or len([i for i in nums if i > 0]) == 1:
        return max_num
    else:
        max_sub_sum = nums[0]
        curr_sub_sum = nums[0]
        for i in nums[1:]:
            curr_sub_sum = i + max(curr_sub_sum, 0)
            max_sub_sum = max(max_sub_sum, curr_sub_sum)
        return max_sub_sum


def maxSubArrayWithIdx(nums: List[int]) -> Tuple[int, int, int]:
    if not nums:
        return 0, -1, -1
    elif len(nums) == 1:
        return nums[0], 0, 0
    elif len(nums) == 2:
        max_num = max(*nums, sum(nums))
        if max_num in nums:
            num_idx = nums.index(max_num)
            return max_num, num_idx, num_idx
        else:
            return max_num, 0, 1

    max_num = max(nums)
    if max_num <= 0 or len([i for i in nums if i > 0]) == 1:
        num_idx = nums.index(max_num)
        return max_num, num_idx, num_idx
    else:
        l, r = 0, len(nums) - 1
        max_l, max_r = 0, 0
        max_sub_sum = nums[l]
        curr_sub_sum = nums[l]
        for i, n in enumerate(nums[l + 1 :]):
            r = i + 1
            if curr_sub_sum > 0:
                curr_sub_sum += n
            else:
                l = i + 1
                curr_sub_sum = n
            if curr_sub_sum > max_sub_sum:
                max_l, max_r = l, r
                max_sub_sum = curr_sub_sum
        return max_sub_sum, max_l, max_r


def main():
    test_array = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    expected_sum = 6
    expected_idx = (3, 6)
    assert maxSubArray(test_array) == expected_sum, maxSubArrayWithIdx(test_array)
    assert maxSubArrayWithIdx(test_array) == (
        expected_sum,
        *expected_idx,
    ), maxSubArrayWithIdx(test_array)


if __name__ == "__main__":
    main()
