import bisect
from collections import Counter
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
        return "a" * (n - 1) + "b"

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
                if equation[j - 1] == 'x':
                    x += (1 if j - 1 == i else int(equation[i:j - 1])) * op
                else:
                    num += int(equation[i:j]) * op
                i = j
        if x == 0:
            return "Infinite solutions" if num == 0 else "No solution"
        return f'x={num // -x}'

    # https://leetcode.cn/problems/maximum-score-after-splitting-a-string/ 分割字符串的最大得分
    def maxScore(self, s: str) -> int:
        ans = score = (s[0] == "0") + s[1:].count("1")
        for c in s[1:-1]:
            score += (1 if c == "0" else -1)
            ans = max(ans, score)
        return ans

    # https://leetcode.cn/problems/maximum-equal-frequency/ 最大相等频率
    def maxEqualFreq(self, nums: List[int]) -> int:
        count, freq = Counter(), Counter()
        ans = max_freq = 0
        for i, num in enumerate(nums):
            if count[num]:
                freq[count[num]] -= 1
            count[num] += 1
            freq[count[num]] += 1
            max_freq = max(max_freq, count[num])
            if max_freq == 1 \
                    or freq[max_freq] * max_freq == i \
                    or (freq[max_freq - 1] * (max_freq - 1) + freq[max_freq] * max_freq == i + 1
                        and freq[max_freq] == 1):
                ans = max(ans, i + 1)
        return ans

    # https://leetcode.cn/problems/number-of-students-doing-homework-at-a-given-time/ 在既定时间做作业的学生人数
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        max_end_time = max(endTime)
        if queryTime > max_end_time:
            return 0
        dif = [0] * (max_end_time + 2)
        for st, ed in zip(startTime, endTime):
            dif[st] += 1
            dif[ed + 1] -= 1
        return sum(dif[:queryTime + 1])

    # https://leetcode.cn/problems/make-two-arrays-equal-by-reversing-sub-arrays/ 通过翻转子数组使两个数组相等
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return Counter(target) == Counter(arr)

    # https://leetcode.cn/problems/find-k-closest-elements/ 找到 K 个最接近的元素
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        arr.sort(key=lambda a: abs(a - x))
        return sorted(arr[:k])

    # https://leetcode.cn/problems/maximum-product-of-two-elements-in-an-array/ 数组中两元素的最大乘积
    def maxProduct(self, nums: List[int]) -> int:
        nums.sort()
        return (nums[-1] - 1) * (nums[-2] - 1)

    # https://leetcode.cn/problems/maximum-width-of-binary-tree/ 二叉树最大宽度
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 1
        que = [[root, 1]]
        while que:
            nxt = []
            for node, idx in que:
                if node.left:
                    nxt.append([node.left, 2 * idx])
                if node.right:
                    nxt.append([node.right, 2 * idx + 1])
            ans = max(ans, que[-1][1] - que[0][1] + 1)
            que = nxt
        return ans

    # https://leetcode.cn/problems/shuffle-the-array/ 重新排列数组
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        ans = []
        x, y = 0, n
        for _ in range(n):
            ans.append(nums[x])
            ans.append(nums[y])
            x += 1
            y += 1
        return ans

    def shuffle_1(self, nums: List[int], n: int) -> List[int]:
        ans = [0] * (2 * n)
        for i in range(n):
            ans[2 * i] = nums[i]
            ans[2 * i + 1] = nums[n + i]
        return ans

    # https://leetcode.cn/problems/validate-stack-sequences/ 验证栈序列
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        if len(pushed) - len(popped):
            return False
        queue = []
        i = 0
        for num in pushed:
            queue.append(num)
            while queue and queue[-1] == popped[i]:
                queue.pop(-1)
                i += 1
        return not queue

    # https://leetcode.cn/problems/final-prices-with-a-special-discount-in-a-shop/ 商品折扣后的最终价格
    def finalPrices(self, prices: List[int]) -> List[int]:
        if not prices:
            return prices
        stack, n = [0], len(prices)
        ans = [0] * n
        for i in range(n - 1, -1, -1):
            p = prices[i]
            while stack and stack[-1] > p:
                stack.pop()
            ans[i] = prices[i] - stack[-1]
            stack.append(p)
        return ans

    # https://leetcode.cn/problems/maximum-length-of-pair-chain/ 最长数对链
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort()
        dp = [1] * len(pairs)
        for i, pair in enumerate(pairs):
            for j in range(i):
                if pair[0] > pairs[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return dp[-1]

    def findLongestChain_1(self, pairs: List[List[int]]) -> int:
        ans = []
        pairs.sort()
        for pair in pairs:
            i = bisect.bisect_left(ans, pair[0])
            if i < len(ans):
                ans[i] = min(ans[i], pair[1])
            else:
                ans.append(pair[1])
        return len(ans)

    # https://leetcode.cn/problems/special-positions-in-a-binary-matrix/ 二进制矩阵中的特殊位置
    def numSpecial(self, mat: List[List[int]]) -> int:
        row_sum = [sum(i) for i in mat]
        col_sum = [sum(i) for i in zip(*mat)]
        ans = 0
        for i, nums in enumerate(mat):
            for j, num in enumerate(nums):
                if num and row_sum[i] and col_sum[j]:
                    ans += 1
        return ans

    def numSpecial_1(self, mat: List[List[int]]) -> int:
        for i, nums in enumerate(mat):
            val = sum(nums) - (i == 0)
            if val:
                for j, num in enumerate(nums):
                    if num:
                        mat[0][j] += val
        return sum(i == 1 for i in mat[0])


# https://leetcode.cn/problems/design-an-ordered-stream/ 设计有序流
class OrderedStream:

    def __init__(self, n: int):
        self.stream = [""] * (n + 1)
        self.ptr = 1

    def insert(self, idKey: int, value: str) -> List[str]:
        self.stream[idKey] = value
        ans = []
        while self.ptr < len(self.stream) and self.stream[self.ptr]:
            ans.append(self.stream[self.ptr])
            self.ptr += 1
        return ans


if __name__ == '__main__':
    s = Solution()
    s.intersectionSizeTwo([[1, 2], [2, 3], [2, 4], [4, 5]])
    print(500 + 55100 + 678 + 1500 + 5000 + 43000 + 1026 + 7280 + 20000 + 580 + 430 + 1932 + 6800)
