import bisect
import collections
import heapq
from collections import Counter
from functools import reduce, lru_cache
from itertools import product
from math import inf, gcd
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
            if arr[i:i + len(l)] != l:
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
            if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
                a, b = at, bt + 1
            if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
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

    # https://leetcode.cn/problems/sign-of-the-product-of-an-array/ 数组元素积的符号
    def arraySign(self, nums: List[int]) -> int:
        ans = 1
        for i in nums:
            if i == 0:
                return 0
            elif i < 0:
                ans *= -1
        return ans

    # https://leetcode.cn/problems/magical-string/ 神奇字符串
    def magicalString(self, n: int) -> int:
        if n < 4:
            return 1
        array = [0] * n
        array[0] = 1
        array[1] = array[2] = 2
        i, j, ans = 2, 2, 1
        while j < n - 1:
            k = array[i]
            num = 3 - array[j]
            while k and j < n - 1:
                if num == 1:
                    ans += 1
                array[j] = num
                j += 1
                k -= 1
            i += 1
        return ans

    # https://leetcode.cn/problems/check-if-two-string-arrays-are-equivalent/ 检查两个字符串是否相等
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        f1 = f2 = 0
        l1 = l2 = 0
        m, n = len(word1), len(word2)
        while f1 < m and f2 < n:
            if word1[f1][l1] != word2[f2][l2]:
                return False
            l1 += 1
            if l1 == len(word1[f1]):
                l1 = 0
                f1 += 1
            l2 += 1
            if l2 == len(word2[f2]):
                l2 = 0
                f2 += 1
        return f1 == m and f2 == n

    # https://leetcode.cn/problems/coordinate-with-maximum-network-quality/网络信号最好的坐标
    def bestCoordinate(self, towers: List[List[int]], radius: int) -> List[int]:
        max_row = max([tower[0] for tower in towers])
        max_col = max([tower[1] for tower in towers])
        x = y = z = 0
        for i in range(max_row + 1):
            for j in range(max_col + 1):
                cval = 0
                for tower in towers:
                    d = (i - tower[0]) ** 2 + (j - tower[1]) ** 2
                    if d <= radius ** 2:
                        cval += tower[2] // (1 + d ** 0.5)
                if cval > z:
                    x, y, z = i, j, cval
        return [x, y]

    # https://leetcode.cn/problems/reach-a-number/ 到达终点数字
    def reachNumber(self, target: int) -> int:
        target = abs(target)
        k = 0
        while target:
            k += 1
            target -= k
        return k if target & 1 == 0 else k + 1 + (k & 1)

    # https://leetcode.cn/problems/count-the-number-of-consistent-strings/ 统计一致字符串的数目
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        a = ans = 0
        for c in allowed:
            a |= 1 << (ord(c) - ord('a'))
        for word in words:
            f = True
            for w in word:
                if a | (1 << (ord(w) - ord('a'))) != a:
                    f = False
                    break
            if f:
                ans += 1
        return ans

    # https://leetcode.cn/problems/ambiguous-coordinates/ 模糊坐标
    def ambiguousCoordinates(self, s: str) -> List[str]:
        def get_position(st: str) -> List[str]:
            ans = []
            if st[0] != '0' or st == '0':
                ans.append(st)
            if st[-1] == '0':
                return ans
            n, f = len(st), False
            for i in range(1, n):
                if f:
                    break
                if st[0] == '0':
                    f = True
                ans.append(f"{st[:i]}.{st[i:]}")
            return ans

        s = s[1:len(s) - 1]
        l = len(s)
        res = []
        for idx in range(1, l):
            ls = get_position(s[:idx])
            if not ls:
                continue
            rs = get_position(s[idx:])
            if not rs:
                continue
            for lsv, rsv in product(ls, rs):
                res.append(f"({lsv}, {rsv})")
        return res

    # https://leetcode.cn/problems/global-and-local-inversions/ 全局倒置和局部倒置
    def isIdealPermutation(self, nums: List[int]) -> bool:
        if not nums:
            return True
        min_val = nums[-1]
        n = len(nums)
        for i in range(n - 3, -1, -1):
            if nums[i] > min_val:
                return False
            min_val = min(min_val, nums[i + 1])
        return True

    # https://leetcode.cn/problems/number-of-matching-subsequences/ 匹配子序列的单词数
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        d = collections.defaultdict(list)
        res = 0
        for i, word in enumerate(words):
            d[word[0]].append((i, 0))
        for c in s:
            p = d[c]
            if not p:
                continue
            d[c] = []
            for i, j in p:
                if j == len(words[i]) - 1:
                    res += 1
                else:
                    d[words[i][j + 1]].append((i, j + 1))
        return res

    # https://leetcode.cn/problems/maximum-number-of-balls-in-a-box/ 盒子中小球的最大数量
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        cnt = Counter(sum(map(int, str(i))) for i in range(lowLimit, highLimit + 1))
        return max(cnt.values())

    # https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/ 区间子数组个数
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        ans = 0
        l1 = l2 = -1
        for i, num in enumerate(nums):
            if left <= num <= right:
                l1 = i
            elif num > right:
                l1, l2 = -1, i
            if l1 != -1:
                ans += l1 - l2
        return ans

    # https://leetcode.cn/problems/minimum-changes-to-make-alternating-binary-string/ 生成交替二进制字符串的最少操作数
    def minOperations(self, s: str) -> int:
        a = sum(int(v) != i & 1 for i, v in enumerate(s))
        return min(a, len(s) - a)

    # https://leetcode.cn/problems/number-of-different-integers-in-a-string/ 字符串中不同整数的数目
    def numDifferentIntegers(self, word: str) -> int:
        n = len(word)
        a = b = 0
        se = set()
        while a < n:
            if not word[a].isdigit():
                a += 1
                continue
            b = a + 1
            while b < n and word[b].isdigit():
                b += 1
            while a < b - 1 and word[a] == '0':
                a += 1
            se.add(word[a:b])
            a = b
        return len(se)

    # https://leetcode.cn/problems/sum-of-beauty-of-all-substrings/ 所有子字符串美丽值之和
    def beautySum(self, s: str) -> int:
        if not s:
            return -1
        res = 0
        n = len(s)
        for i in range(n):
            cnt = Counter()
            mx = 0
            for j in range(i, n):
                cnt[s[j]] += 1
                mx = max(mx, cnt[s[j]])
                mn = min(cnt.values())
                res += mx - mn
        return res

    # https://leetcode.cn/problems/sum-of-digits-of-string-after-convert/ 字符串转化后的各位数字之和
    def getLucky(self, s: str, k: int) -> int:
        if not s:
            return -1
        st = "".join(str(ord(c) - ord('a') + 1) for c in s)
        ans = 0
        for _ in range(k):
            ans = sum(int(c) for c in st)
            st = str(ans)
            if len(st) == 1:
                break
        return ans

    # https://leetcode.cn/problems/minimum-elements-to-add-to-form-a-given-sum/ 构成特定和需要添加的最少元素
    def minElements(self, nums: List[int], limit: int, goal: int) -> int:
        return (abs(goal - sum(nums)) + limit - 1) // limit

    # https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/ 袋子里最少数目的球
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        l, r, ans = 1, max(nums), 0
        while l <= r:
            m = (l + r) // 2
            ops = sum((num - 1) // m for num in nums)
            if ops <= maxOperations:
                ans = m
                r = m - 1
            else:
                l = m + 1
        return ans

    # https://leetcode.cn/problems/maximum-score-from-removing-stones/ 移除石子的最大得分
    def maximumScore(self, a: int, b: int, c: int) -> int:
        arr = sorted([a, b, c])
        if arr[0] + arr[1] <= arr[2]:
            return arr[0] + arr[1]
        return (arr[0] + arr[1] + arr[2]) // 2

    # https://leetcode.cn/problems/minimum-length-of-string-after-deleting-similar-ends/ 删除字符串两端相同字符后的最短长度
    def minimumLength(self, s: str) -> int:
        a, b = 0, len(s) - 1
        while a < b and s[a] == s[b]:
            c = s[a]
            while a <= b and s[a] == c:
                a += 1
            while b > a and s[b] == c:
                b -= 1
        return b - a + 1

    # https://leetcode.cn/problems/check-if-numbers-are-ascending-in-a-sentence/ 检查句子中的数字是否递增
    def areNumbersAscending(self, s: str) -> bool:
        prev = i = 0
        while i < len(s):
            if s[i].isdigit():
                d = 0
                while i < len(s) and s[i].isdigit():
                    d = d * 10 + int(s[i])
                    i += 1
                if d <= prev:
                    return False
                prev = d
            else:
                i += 1
        return True

    # https://leetcode.cn/problems/count-integers-with-even-digit-sum/ 统计各位数字之和为偶数的整数个数
    def countEven(self, num: int) -> int:
        return num // 10 * 5 + (num % 10 + 2 - (sum(map(int, str(num // 10))) & 1)) // 2 - 1

    # https://leetcode.cn/problems/minimum-number-of-operations-to-reinitialize-a-permutation/ 还原排列的最少操作步数
    def reinitializePermutation(self, n: int) -> int:
        origin = list(range(n))
        target = origin.copy()
        step = 0
        while True:
            step += 1
            origin = [origin[n // 2 + (i - 1) // 2] if i & 1 else origin[i // 2] for i in range(n)]
            if origin == target:
                return step

    # https://leetcode.cn/problems/check-if-number-has-equal-digit-count-and-digit-value/ 判断一个数的数字计数是否等于数位的值
    def digitCount(self, num: str) -> bool:
        c = Counter(num)
        for idx, st in enumerate(num):
            if c[str(idx)] != int(st):
                return False
        return True

    # https://leetcode.cn/problems/evaluate-the-bracket-pairs-of-a-string/ 替换字符串中的括号内容
    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        d = dict(knowledge)
        ans, start = [], -1
        for i, c in enumerate(s):
            if c == "(":
                start = i
            elif c == ")":
                ans.append(d.get(s[start + 1:i], "?"))
                start = -1
            elif start == -1:
                ans.append(c)
        return "".join(ans)

    # https://leetcode.cn/problems/rearrange-characters-to-make-target-string/ 重排字符形成目标字符串
    def rearrangeCharacters(self, s: str, target: str) -> int:
        ans = inf
        cnt_s = Counter(s)
        cnt_target = Counter(target)
        for c, cnt in cnt_target.items():
            if not (ans := min(ans, cnt_s[c] // cnt)):
                return 0
        return ans

    # https://leetcode.cn/problems/check-if-matrix-is-x-matrix/ 判断矩阵是否是一个 X 矩阵
    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                if (x == 0) == (i == j or (i + j) == len(grid) - 1):
                    return False
        return True

    # https://leetcode.cn/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/
    # 警告一小时内使用相同员工卡大于等于三次的人
    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        time_map = collections.defaultdict(list)
        for name, time in zip(keyName, keyTime):
            time_map[name].append(int(time[:2]) * 60 + int(time[3:]))
        ans = []
        for name, time_list in time_map.items():
            time_list.sort()
            if any(t2 - t1 <= 60 for t1, t2 in zip(time_list, time_list[2:])):
                ans.append(name)
        ans.sort()
        return ans

    # https://leetcode.cn/problems/remove-sub-folders-from-the-filesystem/ 删除子文件夹
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        folder.sort()
        ans = [folder[0]]
        for i in range(1, len(folder)):
            if not ((pre := len(ans[-1])) < len(folder[i]) and ans[-1] == folder[i][:pre] and folder[i][pre] == '/'):
                ans.append(folder[i])
        return ans

    # https://leetcode.cn/problems/best-poker-hand/ 最好的扑克手牌
    def bestHand(self, ranks: List[int], suits: List[str]) -> str:
        if len(set(suits)) == 1:
            return "Flush"
        c = Counter(ranks)
        if len(c) == 5:
            return "High Card"
        for v in c.values():
            if v > 2:
                return "Three of a Kind"
        return "Pair"

    # https://leetcode.cn/problems/make-array-zero-by-subtracting-equal-amounts/ 使数组中所有元素都等于零
    def minimumOperations(self, nums: List[int]) -> int:
        return len(set(nums) - {0})

    # https://leetcode.cn/problems/minimum-swaps-to-make-strings-equal/ 交换字符使得字符串相同
    def minimumSwap(self, s1: str, s2: str) -> int:
        xy = yx = 0
        for a, b in zip(s1, s2):
            if a == 'x' and b == 'y':
                xy += 1
            elif a == 'y' and b == 'x':
                yx += 1
        if (xy + yx) & 1:
            return -1
        return xy // 2 + yx // 2 + xy % 2 + yx % 2

    # https://leetcode.cn/problems/decrease-elements-to-make-array-zigzag/ 递减元素使数组呈锯齿状
    def movesToMakeZigzag(self, nums: List[int]) -> int:
        even = odd = 0
        for i, num in enumerate(nums):
            a = 0
            if i - 1 >= 0:
                a = max(a, nums[i] - nums[i - 1] + 1)
            if i + 1 < len(nums):
                a = max(a, nums[i] - nums[i + 1] + 1)
            if i & 1:
                odd += a
            else:
                even += a
        return min(even, odd)

    # https://leetcode.cn/problems/merge-similar-items/ 合并相似的物品
    def mergeSimilarItems(self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
        m = Counter()
        for v, w in items1:
            m[v] += w
        for v, w in items2:
            m[v] += w
        return sorted([[v, w] for v, w in m.items()])

    # https://leetcode.cn/problems/largest-local-values-in-a-matrix/ 矩阵中的局部最大值
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        ans = [[0] * (n - 2) for _ in range(n - 2)]
        for i in range(n - 2):
            for j in range(n - 2):
                ans[i][j] = max(grid[a][b] for a in range(i, i + 3) for b in range(j, j + 3))
        return ans

    # https://leetcode.cn/problems/making-file-names-unique/ 保证文件名唯一
    def getFolderNames(self, names: List[str]) -> List[str]:
        ans = []
        m = {}
        for name in names:
            if name not in m:
                ans.append(name)
                m[name] = 1
            else:
                k = m[name]
                while (s := f"{name}({k})") in m:
                    k += 1
                ans.append(s)
                m[s] = 1
                m[name] = k
        return ans

    # https://leetcode.cn/problems/bianry-number-to-string-lcci/ 面试题 05.02. 二进制数转字符串
    def printBin(self, num: float) -> str:
        ans = "0."
        while len(ans) <= 32 and num:
            num *= 2
            b = int(num)
            ans += str(b)
            num -= b
        return ans if len(ans) <= 32 else 'ERROR'

    # https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/ 按位与为零的三元组
    def countTriplets(self, nums: List[int]) -> int:
        ans = 0
        a = Counter(x & y for x in nums for y in nums)
        for num in nums:
            for x, cnt in a.items():
                if not (num & x):
                    ans += cnt
        return ans

    # https://leetcode.cn/problems/minimum-deletions-to-make-string-balanced/ 使字符串平衡的最少删除次数
    def minimumDeletions(self, s: str) -> int:
        lb, ra = 0, s.count('a')
        ans = ra
        for c in s:
            if c == 'a':
                ra -= 1
            else:
                lb += 1
            ans = min(ans, lb + ra)
        return ans

    # https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/ 礼物的最大价值
    def maxValue(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i > 0:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j])
                if j > 0:
                    dp[i][j] = max(dp[i][j], dp[i][j - 1])
                dp[i][j] += grid[i][j]
        return dp[m - 1][n - 1]

    # https://leetcode.cn/problems/minimum-recolors-to-get-k-consecutive-black-blocks/ 得到 K 个黑块的最少涂色次数
    def minimumRecolors(self, blocks: str, k: int) -> int:
        ans, op = inf, 0
        for i, b in enumerate(blocks):
            if b == 'W':
                op += 1
            if i >= k and blocks[i - k] == 'W':
                op -= 1
            if i >= k - 1:
                ans = min(ans, op)
        return ans

    # https://leetcode.cn/problems/find-longest-subarray-lcci/ 字母与数字
    def findLongestSubarray(self, array: List[str]) -> List[str]:
        m = {0: -1}
        l = s = idx = 0
        for i, v in enumerate(array):
            if v.isdigit():
                s += 1
            else:
                s -= 1
            if s in m:
                cl = i - m[s]
                if cl > l:
                    l, idx = cl, m[s] + 1
            else:
                m[s] = i
        if l == 0:
            return []
        return array[idx:idx + l]

    # https://leetcode.cn/problems/make-sum-divisible-by-p/ 使数组和能被 P 整除
    def minSubarray(self, nums: List[int], p: int) -> int:
        s = sum(nums)
        x = s % p
        if not x:
            return 0
        m = {0: -1}
        cs, ans = 0, len(nums)
        for i, v in enumerate(nums):
            cs = (cs + v) % p
            if (t := (cs - x) % p) in m:
                ans = min(ans, i - m[t])
            m[cs] = i
        return -1 if ans == len(nums) else ans

    # https://leetcode.cn/problems/minimum-hours-of-training-to-win-a-competition/ 赢得比赛需要的最少训练时长
    def minNumberOfHours(self, initialEnergy: int, initialExperience: int, energy: List[int],
                         experience: List[int]) -> int:
        total_energy = sum(energy)
        ans = 0 if initialEnergy > total_energy else 1 + total_energy - initialEnergy
        for e in experience:
            if initialExperience > e:
                initialExperience += e
            else:
                ans += (1 + e - initialExperience)
                initialExperience = 2 * e + 1
        return ans

    # https://leetcode.cn/problems/find-valid-matrix-given-row-and-column-sums/ 给定行和列的和求可行矩阵
    def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
        m, n = len(rowSum), len(colSum)
        ans = [[0] * n for _ in range(m)]
        i = j = 0
        while i < m and j < n:
            ans[i][j] = min(rowSum[i], colSum[j])
            rowSum[i] -= ans[i][j]
            colSum[j] -= ans[i][j]
            if not rowSum[i]:
                i += 1
            if not colSum[j]:
                j += 1
        return ans

    # https://leetcode.cn/problems/maximal-network-rank/ 最大网络秩
    def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
        rc = [0] * n
        con = [[0] * n for _ in range(n)]
        for a, b in roads:
            con[a][b] = con[b][a] = True
            rc[a] += 1
            rc[b] += 1
        max_val = 0
        for i in range(n):
            for j in range(i + 1, n):
                val = rc[i] + rc[j] - con[i][j]
                max_val = max(max_val, val)
        return max_val

    # https://leetcode.cn/problems/count-subarrays-with-median-k/ 统计中位数为 K 的子数组
    def countSubarrays(self, nums: List[int], k: int) -> int:
        def trans(n: int) -> int:
            if not n:
                return 0
            return -1 if n < 0 else 1

        idx = nums.index(k)
        c = Counter()
        c[0] = 1
        ans = s = 0
        for i, v in enumerate(nums):
            s += trans(v - k)
            if i < idx:
                c[s] += 1
            else:
                ans += (c[s] + c[s - 1])
        return ans

    # https://leetcode.cn/problems/convert-the-temperature/ 温度转换
    def convertTemperature(self, celsius: float) -> List[float]:
        return [celsius + 273.15, celsius * 1.8 + 32]

    # https://leetcode.cn/problems/best-team-with-no-conflicts/ 无矛盾的最佳球队
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        ans, n = 0, len(scores)
        dp = [0] * n
        t = sorted(zip(scores, ages))
        for i in range(n):
            for j in range(i):
                if t[i][1] >= t[j][1]:
                    dp[i] = max(dp[i], dp[j])
            dp[i] += t[i][0]
            ans = max(ans, dp[i])
        return ans

    # https://leetcode.cn/problems/arithmetic-subarrays/ 等差子数组
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        ans = []
        for s, e in zip(l, r):
            minv = min(nums[s:e + 1])
            maxv = max(nums[s:e + 1])
            if minv == maxv:
                ans.append(True)
                continue
            d, f = divmod(maxv - minv, e - s)
            if f:
                ans.append(False)
                continue
            sl, flag = set(), True
            for i in range(s, e + 1):
                t, tf = divmod(nums[i] - minv, d)
                if tf or (t in sl):
                    flag = False
                    break
                sl.add(t)
            ans.append(flag)
        return ans

    # https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/ 删除最短的子数组使剩余数组有序
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        n = len(arr)
        j = n - 1
        while j > 0 and arr[j - 1] <= arr[j]:
            j -= 1
        if not j:
            return 0
        ans = j
        for i in range(n):
            while j < n and arr[j] < arr[i]:
                j += 1
            ans = min(ans, j - 1 - i)
            if i < n - 1 and arr[i + 1] < arr[i]:
                break
        return ans

    # https://leetcode.cn/problems/find-subarrays-with-equal-sum/ 和相等的子数组
    def findSubarrays(self, nums: List[int]) -> bool:
        n = len(nums)
        s = set()
        for i in range(n - 1):
            c = nums[i] + nums[i + 1]
            if c in s:
                return True
            s.add(c)
        return False

    # https://leetcode.cn/problems/count-sorted-vowel-strings/ 统计字典序元音字符串的数目
    def countVowelStrings(self, n: int) -> int:
        dp = [1] * 5
        for _ in range(n - 1):
            for i in range(1, 5):
                dp[i] += dp[i - 1]
        return sum(dp)

    # https://leetcode.cn/problems/number-of-arithmetic-triplets/ 算术三元组的数目
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        s = set(nums)
        ans = 0
        for i in nums:
            if (i + diff) in s and (i + 2 * diff) in s:
                ans += 1
        return ans

    # https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/ 多边形三角剖分的最低得分
    def minScoreTriangulation(self, values: List[int]) -> int:
        @lru_cache(None)
        def dp(i, j):
            if i + 2 > j:
                return 0
            if i + 2 == j:
                return values[i] * values[i + 1] * values[j]
            return min(values[i] * values[j] * values[k] + dp(i, k) + dp(k, j) for k in range(i + 1, j))

        return dp(0, len(values) - 1)

    # https://leetcode.cn/problems/previous-permutation-with-one-swap/ 交换一次的先前排列 3 2 1
    def prevPermOpt1(self, arr: List[int]) -> List[int]:
        n = len(arr)
        for i in range(n - 2, -1, -1):
            if arr[i] > arr[i + 1]:
                j = n - 1
                while arr[j] >= arr[i] or arr[j] == arr[j - 1]:
                    j -= 1
                arr[i], arr[j] = arr[j], arr[i]
                break
        return arr

    # https://leetcode.cn/problems/number-of-common-factors/ 公因子的数目
    def commonFactors(self, a: int, b: int) -> int:
        c, i = gcd(a, b), 1
        ans = 0
        while i ** 2 <= c:
            if not c % i:
                ans += 1
                if i ** 2 != c:
                    ans += 1
            i += 1
        return ans

    # https://leetcode.cn/problems/check-distances-between-same-letters/ 检查相同字母间的距离
    def checkDistances(self, s: str, distance: List[int]) -> bool:
        n = len(s)
        for i in range(n):
            b = distance[ord(s[i]) - ord('a')]
            if b < 0:
                continue
            idx = i + b + 1
            if idx >= n or s[idx] != s[i]:
                return False
            distance[ord(s[i]) - ord('a')] = -1
        return True

    # https://leetcode.cn/problems/next-greater-node-in-linked-list/ 链表中下一个更大的节点
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        ans = []
        stack = []
        idx = 0
        while head:
            ans.append(0)
            while stack and stack[-1][0] < head.val:
                ans[stack[-1][1]] = head.val
                stack.pop(-1)
            stack.append((head.val, idx))
            idx += 1
            head = head.next
        return ans

    # https://leetcode.cn/problems/add-two-numbers/ 两数相加
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = curr = ListNode()
        ext = 0
        while l1 or l2 or ext:
            val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + ext
            curr.next = ListNode(val % 10)
            ext = val // 10
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        return head.next

    # https://leetcode.cn/problems/add-two-numbers-ii/ 两数相加 II
    def addTwoNumbers2(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        ext, prev = 0, None
        while s1 or s2 or ext:
            val = (s1.pop() if s1 else 0) + (s2.pop() if s2 else 0) + ext
            curr = ListNode(val % 10)
            curr.next = prev
            prev = curr
            ext = val // 10
        return prev

    # https://leetcode.cn/problems/sum-in-a-matrix/ 矩阵中的和
    def matrixSum(self, nums: List[List[int]]) -> int:
        m, n = len(nums), len(nums[0])
        ans = 0
        pq = []
        for i in range(m):
            pq.append([])
            for j in range(n):
                heapq.heappush(pq[i], -nums[i][j])
        for i in range(n):
            max_val = 0
            for j in range(m):
                max_val = max(max_val, -heapq.heappop(pq[j]))
            ans += max_val
        return ans

    # https://leetcode.cn/problems/k-items-with-the-maximum-sum/ K 件物品的最大和
    def kItemsWithMaximumSum(self, numOnes: int, numZeros: int, numNegOnes: int, k: int) -> int:
        return min(k, numOnes) - max(0, k - numOnes - numZeros)

    # https://leetcode.cn/problems/maximum-split-of-positive-even-integers/ 拆分成最多数目的正偶数之和
    def maximumEvenSplit(self, finalSum: int) -> List[int]:
        if finalSum & 1:
            return []
        i, ans = 2, []
        while i <= finalSum:
            ans.append(i)
            finalSum -= i
            i += 2
        ans[-1] += finalSum
        return ans

    # https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/ 两数之和 II - 输入有序数组
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        while i < j:
            val = numbers[i] + numbers[j]
            if val == target:
                return [i + 1, j + 1]
            elif val < target:
                i += 1
            else:
                j -= 1
        return [-1, -1]

    # https://leetcode.cn/problems/3sum/ 三数之和
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if nums is None or len(nums) < 3:
            return []
        ans, n = [], len(nums)
        nums.sort()
        for i in range(n - 2):
            if (nums[i] + nums[i + 1] + nums[i + 2]) > 0:
                break
            if (i and nums[i] == nums[i - 1]) or (nums[i] + nums[n - 2] + nums[n - 1]) < 0:
                continue
            target, k = -nums[i], n - 1
            for j in range(i + 1, n - 1):
                if j != i + 1 and nums[j] == nums[j - 1]:
                    continue
                while j < k and nums[j] + nums[k] > target:
                    k -= 1
                if j == k:
                    break
                if nums[j] + nums[k] == target:
                    ans.append([nums[i], nums[j], nums[k]])
        return ans

    # https://leetcode.cn/problems/3sum-closest/ 最接近的三数之和
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if nums is None or len(nums) < 3:
            return -1
        n, ans = len(nums), 10 ** 7
        nums.sort()

        def update_val(curr):
            nonlocal ans
            if abs(curr - target) < abs(ans - target):
                ans = curr

        for i in range(n - 2):
            if i and nums[i] == nums[i - 1]:
                continue
            if (v := nums[i] + nums[i + 1] + nums[i + 2]) > target:
                update_val(v)
                return ans
            if (v := nums[i] + nums[n - 1] + nums[n - 2]) < target:
                update_val(v)
                continue
            j, k = i + 1, n - 1
            while j < k:
                v = nums[i] + nums[j] + nums[k]
                if v == target:
                    return target
                update_val(v)
                if v < target:
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                elif v > target:
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans

    # https://leetcode.cn/problems/maximum-alternating-subsequence-sum/ 最大子序列交替和
    def maxAlternatingSum(self, nums: List[int]) -> int:
        even, odd = nums[0], 0
        for i in range(1, len(nums)):
            even, odd = max(even, odd + nums[i]), max(odd, even - nums[i])
        return even

    # https://leetcode.cn/problems/alternating-digit-sum/ 交替数字和
    def alternateDigitSum(self, n: int) -> int:
        ans, flag = 0, 1
        while n:
            ans += n % 10 * flag
            flag *= -1
            n //= 10
        return -flag * ans

    # https://leetcode.cn/problems/minimum-falling-path-sum/ 下降路径最小和
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        dp = matrix[0]
        m, n = len(matrix), len(dp)
        for i in range(1, m):
            curr = dp.copy()
            for j in range(n):
                if j > 0:
                    curr[j] = min(curr[j], dp[j - 1])
                if j < m - 1:
                    curr[j] = min(curr[j], dp[j + 1])
                curr[j] += matrix[i][j]
            dp = curr
        return min(dp)

    # https://leetcode.cn/problems/4sum/ 四数之和
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if not nums or len(nums) < 4:
            return []
        n, ans = len(nums), []
        nums.sort()
        for i in range(n - 3):
            if i and nums[i] == nums[i - 1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[n - 1] + nums[n - 2] + nums[n - 3] < target:
                continue
            for j in range(i + 1, n - 2):
                if j != i + 1 and nums[j] == nums[j - 1]:
                    continue
                if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    break
                if nums[i] + nums[j] + nums[n - 1] + nums[n - 2] < target:
                    continue
                k, l = j + 1, n - 1
                while k < l:
                    val = nums[i] + nums[j] + nums[k] + nums[l]
                    if val == target:
                        ans.append([nums[i], nums[j], nums[k], nums[l]])
                        k += 1
                        while k < j and nums[k] == nums[k - 1]:
                            k += 1
                        l -= 1
                        while k < l and nums[l] == nums[l + 1]:
                            l -= 1
                    elif val < target:
                        k += 1
                    else:
                        l -= 1
        return ans

    # https://leetcode.cn/problems/robot-bounded-in-circle/ 困于环中的机器人
    def isRobotBounded(self, instructions: str) -> bool:
        dict = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        curr = x = y = 0
        for instruction in instructions:
            if instruction == 'G':
                x += dict[curr][0]
                y += dict[curr][1]
            elif instruction == 'L':
                curr = (curr + 3) % 4
            else:
                curr = (curr + 1) % 4
        return curr != 0 or (x == 0 and y == 0)

    # https://leetcode.cn/problems/most-frequent-even-element/ 出现最频繁的偶数元素
    def mostFrequentEven(self, nums: List[int]) -> int:
        cnt = Counter()
        ans, tot = -1, 0
        for num in nums:
            if not (num & 1):
                cnt[num] += 1
                if cnt[num] > tot:
                    tot = cnt[num]
                    ans = num
                elif cnt[num] == tot and num < ans:
                    ans = num
        return ans

    # https://leetcode.cn/problems/count-days-spent-together/ 统计共同度过的日子数
    def countDaysTogether(self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str) -> int:

        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        pre_sum = [0]
        for i in months:
            pre_sum.append(pre_sum[-1] + i)

        def calculate_days(s: str) -> int:
            month = int(s[:2])
            day = int(s[3:])
            return pre_sum[month - 1] + day

        a1 = calculate_days(arriveAlice)
        a2 = calculate_days(leaveAlice)
        b1 = calculate_days(arriveBob)
        b2 = calculate_days(leaveBob)
        return max(0, min(a2, b2) - max(a1, b1) + 1)

    # https://leetcode.cn/problems/partition-array-for-maximum-sum/ 分隔数组以得到最大和
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            max_val = arr[i - 1]
            for j in range(i - 1, max(-1, i - k - 1), -1):
                dp[i] = max(dp[i], dp[j] + max_val * (i - j))
                if j > 0:
                    max_val = max(max_val, arr[j - 1])
        return dp[n]

    # https://leetcode.cn/problems/sort-the-people/ 按身高排序
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        return [name for _, name in sorted(zip(heights, names), reverse=True)]

    # https://leetcode.cn/problems/longest-string-chain/ 最长字符串链
    def longestStrChain(self, words: List[str]) -> int:
        words.sort(key=len)
        ans = 1
        cnt = collections.defaultdict(int)
        for word in words:
            cnt[word] = 1
            for i in range(len(word)):
                tmp = word[:i] + word[i + 1:]
                if tmp in cnt:
                    cnt[word] = max(cnt[word], 1 + cnt[tmp])

            ans = max(cnt[word], ans)
        return ans

    # https://leetcode.cn/problems/check-if-word-is-valid-after-substitutions/ 检查替换后的词是否有效
    def isValid(self, s: str) -> bool:
        t = []
        for c in s:
            t.append(c)
            if ''.join(t[-3:]) == 'abc':
                t[-3:] = []
        return len(t) == 0

    # https://leetcode.cn/problems/powerful-integers/ 强整数
    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        ans = set()
        v1 = 1
        for i in range(21):
            v2 = 1
            for j in range(21):
                v = v1 + v2
                if v <= bound:
                    ans.add(v)
                else:
                    break
                v2 *= y
            if v1 > bound:
                break
            v1 *= x
        return list(ans)

    # https://leetcode.cn/problems/the-employee-that-worked-on-the-longest-task/ 处理用时最长的那个任务的员工
    def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
        idx, time = logs[0]
        for i in range(1, len(logs)):
            c_idx, c_time = logs[i][0], logs[i][1] - logs[i - 1][1]
            if c_time > time or (c_time == time and c_idx < idx):
                idx, time = c_idx, c_time
        return idx

    # https://leetcode.cn/problems/pairs-of-songs-with-total-durations-divisible-by-60/ 总持续时间可被 60 整除的歌曲
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans = 0
        cnt = [0] * 60
        for t in time:
            ans += cnt[(60 - t % 60) % 60]
            cnt[t % 60] += 1
        return ans

    # https://leetcode.cn/problems/number-of-valid-clock-times/ 有效时间的数目 23:59
    def countTime(self, time: str) -> int:
        a, b, c, d = time[0], time[1], time[3], time[4]
        ans = 1
        if d == '?':
            ans *= 10
        if c == '?':
            ans *= 6
        if a == '?' and b == '?':
            ans *= 24
        elif b == '?':
            ans *= 10 if ord(a) < ord('2') else 4
        elif a == '?':
            ans *= 2 if ord(b) > ord('3') else 3
        return ans

    # https://leetcode.cn/problems/flip-columns-for-maximum-number-of-equal-rows/ 按列翻转得到最大值等行数
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        c = Counter()
        for i in range(m):
            v = 0
            for j in range(n):
                v = v * 10 + (matrix[i][j] ^ matrix[i][0])
            c[v] += 1
        return max(c.values())

    # https://leetcode.cn/problems/determine-if-two-events-have-conflict/ 判断两个事件是否存在冲突
    def haveConflict(self, event1: List[str], event2: List[str]) -> bool:
        return not (event1[1] < event2[0] or event2[1] < event1[0])

    # https://leetcode.cn/problems/letter-tile-possibilities/ 活字印刷
    def numTilePossibilities(self, tiles: str) -> int:
        l, c = set(tiles), collections.Counter(tiles)

        def dfs(n: int) -> int:
            if not n:
                return 0
            res = 0
            for t in l:
                if c[t]:
                    res += 1
                    c[t] -= 1
                    res += dfs(n - 1)
                    c[t] += 1
            return res

        return dfs(len(tiles))

    # https://leetcode.cn/problems/o8SXZn/  蓄水
    def storeWater(self, bucket: List[int], vat: List[int]) -> int:
        n = len(bucket)
        max_vat = max(vat)
        if not max_vat:
            return 0
        res = inf
        for i in range(1, max_vat + 1):
            t = 0
            for j in range(n):
                t += max(0, (vat[j] + i - 1) // i - bucket[j])
            res = min(res, t + i)
        return res

    # https://leetcode.cn/problems/largest-values-from-labels/ 受标签影响的最大值
    def largestValsFromLabels(self, values: List[int], labels: List[int], numWanted: int, useLimit: int) -> int:
        tmp = sorted(zip(values, labels), reverse=True)
        ans = cnt = 0
        c = Counter()
        for value, label in tmp:
            if c[label] == useLimit:
                continue
            ans += value
            c[label] += 1
            cnt += 1
            if cnt == numWanted:
                break
        return ans

    # https://leetcode.cn/problems/odd-string-difference/ 差值数组不同的字符串
    def oddString(self, words: List[str]) -> str:
        def get_diff(s: str):
            return [ord(s[i]) - ord(s[i - 1]) for i in range(1, len(s))]

        d0 = get_diff(words[0])
        d1 = get_diff(words[1])
        if d0 == d1:
            for i in range(2, len(words)):
                if get_diff(words[i]) != d0:
                    return words[i]
        return words[1] if get_diff(words[2]) == d0 else words[0]

    # https://leetcode.cn/problems/average-value-of-even-numbers-that-are-divisible-by-three/ 可被三整除的偶数的平均值
    def averageValue(self, nums: List[int]) -> int:
        n = s = 0
        for num in nums:
            if not (num % 6):
                n += 1
                s += num
        return 0 if not n else s // n

    # https://leetcode.cn/problems/apply-operations-to-an-array/ 对数组执行操作
    def applyOperations(self, nums: List[int]) -> List[int]:
        n, j = len(nums), 0
        for i in range(n):
            if i != n - 1 and nums[i] == nums[i + 1]:
                nums[i] *= 2
                nums[i + 1] = 0
            if nums[i]:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
        return nums

    # https://leetcode.cn/problems/equal-row-and-column-pairs/ 相等行列对
    def equalPairs(self, grid: List[List[int]]) -> int:
        c = Counter(tuple(row) for row in grid)
        res = 0
        for colum in zip(*grid):
            res += c[tuple(colum)]
        return res

    # https://leetcode.cn/problems/mice-and-cheese/ 老鼠和奶酪
    def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
        res, n = sum(reward2), len(reward1)
        diff = [reward1[i] - reward2[i] for i in range(n)]
        diff.sort(reverse=True)
        return res + sum(diff[:k])

    # https://leetcode.cn/problems/minimum-cost-tree-from-leaf-values/  叶值的最小代价生成树
    def mctFromLeafValues(self, arr: List[int]) -> int:
        stk = []
        res = 0
        for a in arr:
            while stk and stk[-1] <= a:
                b = stk.pop()
                if not stk or stk[-1] > a:
                    res += a * b
                else:
                    res += b * stk[-1]
            stk.append(a)
        while len(stk) >= 2:
            a = stk.pop()
            res += stk[-1] * a
        return res

    # https://leetcode.cn/problems/compare-strings-by-frequency-of-the-smallest-character/ 比较字符串最小字母出现频次
    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:

        def count(st):
            ch, cnt = 'z', 0
            for a in st:
                if a < ch:
                    ch, cnt = a, 1
                elif a == ch:
                    cnt += 1
            return cnt

        cnts = [0] * 12
        for w in words:
            cnts[count(w)] += 1
        for i in range(9, 0, -1):
            cnts[i] += cnts[i + 1]
        res = []
        for a in queries:
            res.append(cnts[count(a) + 1])
        return res

    # https://leetcode.cn/problems/number-of-unequal-triplets-in-array/ 数组中不等三元组的数目
    def unequalTriplets(self, nums: List[int]) -> int:
        n, prev, res = len(nums), 0, 0
        c = Counter(nums)
        for v in c.values():
            res += prev * v * (n - prev - v)
            prev += v
        return res

    # https://leetcode.cn/problems/number-of-times-binary-string-is-prefix-aligned/ 二进制字符串前缀一致的次数
    def numTimesAllBlue(self, flips: List[int]) -> int:
        res = max_val = 0
        for i, f in enumerate(flips):
            max_val = max(max_val, f)
            if i + 1 == max_val:
                res += 1
        return res

    # https://leetcode.cn/problems/can-make-palindrome-from-substring/ 构建回文串检测
    def canMakePaliQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        res = []
        cnts = [0] * (n + 1)
        for i, c in enumerate(s):
            cnts[i + 1] = cnts[i] ^ (1 << (ord(c) - ord('a')))
        for i, j, k in queries:
            res.append((cnts[i] ^ cnts[j + 1]).bit_count() <= 2 * k + 1)
        return res

    # https://leetcode.cn/problems/minimum-cuts-to-divide-a-circle/ 分割圆的最少切割次数
    def numberOfCuts(self, n: int) -> int:
        if n == 1:
            return 0
        return n if n & 1 else n // 2

    # https://leetcode.cn/problems/greatest-sum-divisible-by-three/ 可被3整除的最大和
    def maxSumDivThree(self, nums: List[int]) -> int:
        a = [num for num in nums if num % 3 == 0]
        b = sorted([num for num in nums if num % 3 == 1], reverse=True)
        c = sorted([num for num in nums if num % 3 == 2], reverse=True)
        lb, lc, res = len(b), len(c), 0
        for i in (lb - 2, lb - 1, lb):
            if i >= 0:
                for j in (lc - 2, lc - 1, lc):
                    if j >= 0 and (i - j) % 3 == 0:
                        res = max(res, sum(b[:i]) + sum(c[:j]))
        return sum(a) + res

    # https://leetcode.cn/problems/circle-and-rectangle-overlapping/ 圆和矩形是否有重叠
    def checkOverlap(self, radius: int, xCenter: int, yCenter: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        des = 0
        if xCenter < x1 or xCenter > x2:
            des += min((x1 - xCenter) ** 2, (x2 - xCenter) ** 2)
        if yCenter < y1 or yCenter > y2:
            des += min((y1 - yCenter) ** 2, (y2 - yCenter) ** 2)
        return des <= radius ** 2

    # https://leetcode.cn/problems/find-the-pivot-integer/ 找出中枢整数
    def pivotInteger(self, n: int) -> int:
        t = (n * n + n) // 2
        res = int(t ** 0.5)
        return -1 if res ** 2 != t else res

    # https://leetcode.cn/problems/maximum-subarray-sum-with-one-deletion/ 删除一次得到子数组最大和
    def maximumSum(self, arr: List[int]) -> int:
        dp0, dp1, res = arr[0], 0, arr[0]
        n = len(arr)
        for i in range(1, n):
            dp1 = max(dp1 + arr[i], dp0)
            dp0 = max(dp0, 0) + arr[i]
            res = max(dp0, dp1, res)
        return res

    # https://leetcode.cn/problems/reconstruct-a-2-row-binary-matrix/ 重构 2 行二进制矩阵
    def reconstructMatrix(self, upper: int, lower: int, colsum: List[int]) -> List[List[int]]:
        sum_val = two_sum = 0
        for c in colsum:
            if c == 2:
                two_sum += 1
            sum_val += c
        if sum_val != upper + lower or two_sum > min(upper, lower):
            return []
        n = len(colsum)
        ans = [[0] * n for _ in range(2)]
        upper -= two_sum
        lower -= two_sum
        for i in range(n):
            if colsum[i] == 2:
                ans[0][i] = ans[1][i] = 1
            elif colsum[i] == 1:
                if upper > 0:
                    ans[0][i] = 1
                    upper -= 1
                else:
                    ans[1][i] = 1
                    lower -= 1
        return ans


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
    print(s.fourSum([1,0,-1,0,-2,2], 0))
