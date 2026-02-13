from typing import List


def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    if n == 0:
        return
    if m == 0:
        for i in range(n):
            nums1[i] = nums2[i]
        return
    if m == n == 1:
        if nums1[0] < nums2[0]:
            nums1[1] = nums2[0]
        else:
            nums1[1] = nums1[0]
            nums1[0] = nums2[0]
        return

    if nums1[m - 1] <= nums2[0]:
        for i, n in enumerate(nums2):
            nums1[m + i] = n
        return

    if nums1[0] >= nums2[-1]:
        for n in reversed(nums2):
            nums1.insert(0, n)
            nums1.pop()
        return

    i, j = 0, 0
    while i < m + n and j < n:
        if nums1[i] > nums2[j]:
            nums1.insert(i, nums2[j])
            nums1.pop()
            j += 1
        i += 1
    while j < n:
        nums1[m + j] = nums2[j]
        j += 1
