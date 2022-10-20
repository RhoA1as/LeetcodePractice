import bisect
import collections
from collections import Counter
from functools import reduce
from operator import or_
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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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

    # https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/ 统计子串中的唯一字符
    def uniqueLetterString(self, s: str) -> int:
        if not s:
            return 0
        d = collections.defaultdict(list)
        ans = 0
        for i, c in enumerate(s):
            d[c].append(i)
        for l in d.values():
            l = [-1] + l + [len(s)]
            for j in range(1, len(l) - 1):
                ans += (l[j] - l[j - 1]) * (l[j + 1] - l[j])
        return ans

    # https://leetcode.cn/problems/rearrange-spaces-between-words/ 重新排列单词间的空格
    def reorderSpaces(self, text: str) -> str:
        space = text.count(" ")
        chars = text.split()
        n = len(chars)
        if n == 1:
            return chars[0] + ' ' * space
        a, b = divmod(space, n - 1)
        return (' ' * a).join(chars) + ' ' * b

    # https://leetcode.cn/problems/beautiful-arrangement-ii/ 优美的排列 II
    def constructArray(self, n: int, k: int) -> List[int]:
        ans = list(range(1, n - k))
        i, j = n - k, n
        while i <= j:
            ans.append(i)
            if i - j:
                ans.append(j)
            i += 1
            j -= 1
        return ans

    # https://leetcode.cn/problems/trim-a-binary-search-tree/ 修剪二叉搜索树
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        while root and (root.val < low or root.val > high):
            if root.val < low:
                root = root.right
            else:
                root = root.left
        if not root:
            return None
        node = root
        while node.left:
            if node.left.val >= low:
                node = node.left
            else:
                node.left = node.left.right
        node = root
        while node.right:
            if node.right.val <= high:
                node = node.right
            else:
                node.right = node.right.left
        return root

    # https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/ 特殊数组的特征值
    def specialArray(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        n = len(nums)
        for i in range(1, n + 1):
            if nums[i - 1] >= i and (i == n or nums[i] < i):
                return i
        return -1

    # https://leetcode.cn/problems/sort-array-by-increasing-frequency/ 按照频率将数组升序排序
    def frequencySort(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        nums.sort(key=lambda x: (cnt[x], -x))
        return nums

    # https://leetcode.cn/problems/k-similar-strings/ 相似度为 K 的字符串
    def kSimilarity(self, s1: str, s2: str) -> int:
        step, n = 0, len(s2)
        queue, visited = [(s1, 0)], {s1}
        while queue:
            size = len(queue)
            for _ in range(size):
                tmp = queue.pop(0)
                s, idx = tmp[0], tmp[1]
                if s == s2:
                    return step
                while s[idx] == s2[idx]:
                    idx += 1
                for j in range(idx + 1, n):
                    if s[j] == s2[idx] and s[j] != s2[j]:
                        l = list(s)
                        l[idx], l[j] = l[j], l[idx]
                        t = "".join(l)
                        if t not in visited:
                            queue.append((t, idx + 1))
                            visited.add(t)
            step += 1

    # https://leetcode.cn/problems/check-array-formation-through-concatenation/ 能否连接形成数组
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        m = {piece[0]: i for i, piece in enumerate(pieces)}
        n, i = len(arr), 0
        while i < n:
            if arr[i] not in m:
                return False
            l = pieces[m[arr[i]]]
            if arr[i:i+len(l)] != l:
                return False
            i += len(l)
        return True

    # https://leetcode.cn/problems/rotated-digits/  旋转数字
    def rotatedDigits(self, n: int) -> int:
        ans = 0
        for i in range(1, n + 1):
            flag = False
            while i:
                j = i % 10
                i //= 10
                if j in (2, 5, 6, 9):
                    flag = True
                elif j in (3, 4, 7):
                    flag = False
                    break
            if flag:
                ans += 1
        return ans

    # https://leetcode.cn/problems/missing-two-lcci/ 消失的两个数字
    def missingTwo(self, nums: List[int]) -> List[int]:
        xorsum = 0
        for i in nums:
            xorsum ^= i
        n = len(nums) + 2
        for i in range(1, n + 1):
            xorsum ^= i
        flag = xorsum & -xorsum
        t1 = t2 = 0
        for i in nums:
            if i & flag:
                t1 ^= i
            else:
                t2 ^= i
        for i in range(1, n + 1):
            if i & flag:
                t1 ^= i
            else:
                t2 ^= i
        return [t1, t2]

    # https://leetcode.cn/problems/check-permutation-lcci/ 判定是否互为字符重排
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        return collections.Counter(s1) == collections.Counter(s2)

    # https://leetcode.cn/problems/get-kth-magic-number-lcci/ 第k个数
    def getKthMagicNumber(self, k: int) -> int:
        p3 = p5 = p7 = 0
        ans = [0] * k
        ans[0] = 1
        for i in range(1, k):
            ans[i] = min(ans[p3] * 3, ans[p5] * 5, ans[p7] * 7)
            if ans[i] == ans[p3] * 3:
                p3 += 1
            if ans[i] == ans[p5] * 5:
                p5 += 1
            if ans[i] == ans[p7] * 7:
                p7 += 1
        return ans[-1]

    # https://leetcode.cn/problems/score-of-parentheses/ 括号的分数
    def scoreOfParentheses(self, s: str) -> int:
        st = [0]
        for c in s:
            if c == '(':
                st.append(0)
            else:
                v = st.pop()
                st[-1] += max(2 * v, 1)
        return st[-1]

    # https://leetcode.cn/problems/minimum-swaps-to-make-sequences-increasing/ 使序列递增的最小交换次数
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        a, b = 0, 1
        for i in range(1, n):
            at, bt = a, b
            a = b = n
            if nums1[i] > nums1[i-1] and nums2[i] > nums2[i-1]:
                a, b = at, bt + 1
            if nums1[i] > nums2[i-1] and nums2[i] > nums1[i-1]:
                a, b = min(a, bt), min(b, at + 1)
        return min(a, b)

    # https://leetcode.cn/problems/check-if-one-string-swap-can-make-strings-equal/ 仅执行一次字符串交换能否使两个字符串相等
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        i = j = -1
        for idx, (a, b) in enumerate(zip(s1, s2)):
            if a == b:
                continue
            if i < 0:
                i = idx
            elif j < 0:
                j = idx
            else:
                return False
        return s1[i] == s2[j] and s1[j] == s2[i]

    # https://leetcode.cn/problems/linked-list-components/ 链表组件
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        ans = 0
        is_head = False
        n = set(nums)
        while head:
            if head.val not in n:
                is_head = False
            elif not is_head:
                ans += 1
                is_head = True
            head = head.next
        return ans

    # https://leetcode.cn/problems/max-chunks-to-make-sorted/ 最多能完成排序的块
    def maxChunksToSorted(self, arr: List[int]) -> int:
        if not arr:
            return 0
        ans = mx = 0
        for i, num in enumerate(arr):
            mx = max(mx, num)
            ans += mx == i
        return ans

    # https://leetcode.cn/problems/number-of-students-unable-to-eat-lunch/ 无法吃午餐的学生数量
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        sum1 = sum(students)
        sum0 = len(students) - sum1
        for sa in sandwiches:
            if sa == 0 and sum0:
                sum0 -= 1
            elif sa == 1 and sum1:
                sum1 -= 1
            else:
                break
        return sum0 + sum1

    # https://leetcode.cn/problems/k-th-symbol-in-grammar/ 第k个语法符号
    def kthGrammar(self, n: int, k: int) -> int:
        return (k - 1).bit_count() & 1


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
