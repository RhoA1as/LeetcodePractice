
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


# https://leetcode.cn/problems/asteroid-collision/ 行星碰撞
def asteroidCollision(self, asteroids: List[int]) -> List[int]:
    if not asteroids:
        return asteroids
    stack = []
    for asteroid in asteroids:
        is_live = True
        while is_live and stack and stack[-1] > 0 and asteroid < 0:
            if stack[-1] >= -asteroid:
                is_live = False
            if stack[-1] <= -asteroid:
                stack.pop()
        if is_live:
            stack.append(asteroid)
    return stack


class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


# https://leetcode.cn/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/ 四叉树交集
def intersect(self, quadTree1: 'Node', quadTree2: 'Node') -> 'Node':
    if quadTree1.isLeaf:
        return Node(True, True) if quadTree1.val else quadTree2
    if quadTree2.isLeaf:
        return self.intersect(quadTree2, quadTree1)
    tl = self.intersect(quadTree1.topLeft, quadTree2.topLeft)
    tr = self.intersect(quadTree1.topRight, quadTree2.topRight)
    bl = self.intersect(quadTree1.bottomLeft, quadTree2.bottomLeft)
    br = self.intersect(quadTree1.bottomRight, quadTree2.bottomRight)
    if tl.isLeaf and tr.isLeaf and bl.isLeaf and br.isLeaf and tl.val == tr.val == bl.val == br.val:
        return Node(tl.val, True)
    return Node(False, False, tl, tr, bl, br)


# https://leetcode.cn/problems/array-nesting/ 数组嵌套
def arrayNesting(self, nums: List[int]) -> int:
    if not nums:
        return 0
    n, ans = len(nums), 0
    for i in range(n):
        if nums[i] == -1:
            continue
        cnt = 0
        while nums[i] != -1:
            tmp = nums[i]
            nums[i] = -1
            i = tmp
            cnt += 1
        ans = max(ans, cnt)
    return ans


