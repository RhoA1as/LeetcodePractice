from typing import List


class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


# https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/ 填充每个节点的下一个右侧节点指针 II
def connect(self, root: 'Node') -> 'Node':
    if not root:
        return root
    queue = [root]
    while queue:
        size = len(queue)
        prev = None
        for _ in range(size):
            cur = queue.pop(0)
            if prev:
                prev.next = cur
            prev = cur
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
    return root


# https://leetcode.cn/problems/find-peak-element/ 寻找峰值
def findPeakElement(self, nums: List[int]) -> int:
    l, r = 0, len(nums) - 1
    while l < r:
        m = l + (r - l) // 2
        if nums[m] < nums[m + 1]:
            l = m + 1
        else:
            r = m
    return l


# https://leetcode.cn/problems/check-if-a-string-is-an-acronym-of-words/ 判别首字母缩略词
def isAcronym(self, words: List[str], s: str) -> bool:
    return len(s) == len(words) and all(words[i][0] == s[i] for i in range(len(words)))


# https://leetcode.cn/problems/find-the-longest-balanced-substring-of-a-binary-string/ 最长平衡子字符串
def findTheLongestBalancedSubstring(self, s: str) -> int:
    res = cnt0 = cnt1 = 0
    for i in range(len(s)):
        if s[i] == "1":
            cnt1 += 1
            res = max(res, 2 * min(cnt0, cnt1))
        elif i == 0 or s[i - 1] == "1":
            cnt0 = 1
            cnt1 = 0
        else:
            cnt0 += 1
    return res


# https://leetcode.cn/problems/longest-even-odd-subarray-with-threshold/ 最长奇偶子数组
def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
    res = i = 0
    n = len(nums)
    while i < n:
        if nums[i] > threshold or nums[i] % 2:
            i += 1
            continue
        start = i
        i += 1
        while i < n and nums[i] <= threshold and nums[i] % 2 != nums[i - 1] % 2:
            i += 1
        res = max(res, i - start)
    return res
