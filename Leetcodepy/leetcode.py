# --*-- coding:utf-8 --*--
# leetcode daily card
from typing import List


class Solution:
    # https://leetcode-cn.com/problems/find-good-days-to-rob-the-bank/ 适合打劫银行的日子
    def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
        n = len(security)
        left = [0] * n
        right = [0] * n
        for i in range(1, n):
            if security[i] <= security[i-1]:
                left[i] = left[i-1] + 1
            if security[n-i-1] <= security[n-i]:
                right[n-i-1] = right[n-i] + 1
        return [i for i in range(time, n-time) if left[i] >= time and right[i] >= time]
