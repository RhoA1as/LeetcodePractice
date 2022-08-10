
from typing import List, Optional


# https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix/ 奇数值单元格的数目
from leetcode import TreeNode


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


# https://leetcode.cn/problems/shift-2d-grid/ 二维网格迁移
def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
    if not grid:
        return grid
    m, n = len(grid), len(grid[0])
    ans = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            idx = (i * n + j + k) % (m * n)
            ans[idx // n][idx % n] = grid[i][j]
    return ans


class Solution:

    # https://leetcode.cn/problems/binary-tree-pruning/ 二叉树减枝
    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if not (root.left or root.right or root.val):
            return None
        return root

    # https://leetcode.cn/problems/set-intersection-size-at-least-two/ 设置交集大小至少为2
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda a: a[1])
        pl, pr, ans = intervals[0][1] - 1, intervals[0][1], 2
        for idx, interval in enumerate(intervals):
            if not idx:
                continue
            if pl >= interval[0] and pr <= interval[1]:
                continue
            if pr < interval[0]:
                ans += 2
                pl, pr = interval[1] - 1, interval[1]
            elif pl < interval[0]:
                ans += 1
                if pr == interval[1]:
                    pl = pr - 1
                else:
                    pl = pr
                    pr = interval[1]
        return ans

    # https://leetcode.cn/problems/rank-transform-of-an-array/ 数组序号转换
    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        mapping = {v: i for i, v in enumerate(sorted(set(arr)), 1)}
        return [mapping[a] for a in arr]

    # https://leetcode.cn/problems/generate-a-string-with-characters-that-have-odd-counts/ 生成每种字符都是奇数个的字符串
    def generateTheString(self, n: int) -> str:
        if n & 1:
            return 'a' * n
        return "a" * (n-1) + "b"

    # https://leetcode.cn/problems/orderly-queue/ 有序队列
    def orderlyQueue(self, s: str, k: int) -> str:
        if k == 1:
            i, j, _k, n = 0, 1, 0, len(s)
            while i < n and j < n and _k < n:
                a, b = s[(i + _k) % n], s[(j + _k) % n]
                if a == b:
                    _k += 1
                else:
                    if a > b:
                        i += _k + 1
                    else:
                        j += _k + 1
                    if i == j:
                        i += 1
                    _k = 0
            i = min(i, j)
            return s[i:] + s[:i]
        return "".join(sorted(s))

    # https://leetcode.cn/problems/minimum-subsequence-in-non-increasing-order/ 非递增顺序的最小子序列
    def minSubsequence(self, nums: List[int]) -> List[int]:
        nums.sort(reverse=True)
        ans, a, s = [], 0, sum(nums)
        for num in nums:
            a += num
            ans.append(num)
            if a > s - a:
                break
        return ans

    # https://leetcode.cn/problems/string-matching-in-an-array/ 数组中的字符串匹配
    def stringMatching(self, words: List[str]) -> List[str]:
        ans = []
        for i, w in enumerate(words):
            for j, o in enumerate(words):
                if i != j and words[i] in words[j]:
                    ans.append(words[i])
                    break
        return ans

    # https://leetcode.cn/problems/exclusive-time-of-functions/ 函数的独占时间
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ans = [0] * n
        stack = []
        for log in logs:
            idx, tpe, timestamp = log.split(":")
            idx, timestamp = int(idx), int(timestamp)
            if tpe == 'start':
                if stack:
                    ans[stack[-1][0]] += timestamp - stack[-1][1]
                stack.append([idx, timestamp])
            else:
                t = stack.pop(-1)
                ans[t[0]] += timestamp - t[1] + 1
                if stack:
                    stack[-1][1] = timestamp + 1
        return ans

    # https://leetcode.cn/problems/solve-the-equation/ 求解方程
    def solveEquation(self, equation: str) -> str:
        if not equation:
            return "No solution"
        i, op, x, num, n = 0, 1, 0, 0, len(equation)
        while i < n:
            if equation[i] == '+':
                op = 1
                i += 1
            elif equation[i] == '-':
                op = -1
                i += 1
            elif equation[i] == '=':
                num *= -1
                x *= -1
                i += 1
                op = 1
            else:
                j = i
                while j < n and equation[j] != '+' and equation[j] != '-' and equation[j] != '=':
                    j += 1
                if equation[j-1] == 'x':
                    x += (1 if j - 1 == i else int(equation[i:j-1])) * op
                else:
                    num += int(equation[i:j]) * op
                i = j
        if x == 0:
            return "Infinite solutions" if num == 0 else "No solution"
        return f'x={num // -x}'


if __name__ == '__main__':
    s = Solution()
    s.intersectionSizeTwo([[1, 2], [2, 3], [2, 4], [4, 5]])
    print(500+55100+678+1500+5000+43000+1026+7280+20000+580+430+1932+6800)
