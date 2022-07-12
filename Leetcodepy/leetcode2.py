
from typing import List


# https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix/ 奇数值单元格的数目
def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
    row = col = 0
    for index in indices:
        row ^= 1 << index[0]
        col ^= 1 << index[1]
    a = b = 0
    for i in range(m):
        if (row >> i) & 1:
            a += 1
    for i in range(n):
        if (col >> i) & 1:
            b += 1
    return a * (n - b) + b * (m - a)
