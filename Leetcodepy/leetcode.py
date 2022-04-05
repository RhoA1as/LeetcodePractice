# --*-- coding:utf-8 --*--
# leetcode daily card
import collections
from functools import reduce
from heapq import heappop, heappush
from itertools import product
from math import inf
from operator import or_
from typing import List, Optional
from sortedcontainers import SortedList


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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

    # https://leetcode-cn.com/problems/base-7/ 七进制数
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return '0'
        negative = num < 0
        num = abs(num)
        ans = []
        while num:
            ans.append(str(num % 7))
            num //= 7
        if negative:
            ans.append("-")
        return "".join(reversed(ans))

    # https://leetcode-cn.com/problems/plates-between-candles/ 蜡烛之间的盘子
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        if not s:
            return []
        n = len(s)
        pre_sum, left, right, ans = [], [0] * n, [0] * n, []
        val, l, r = 0, -1, -1
        for i in range(n):
            if s[i] == '*':
                val += 1
            pre_sum.append(val)
            if s[i] == '|':
                l = i
            left[i] = l
            if s[n-i-1] == '|':
                r = n-i-1
            right[n-i-1] = r
        for i, j in queries:
            x, y = right[i], left[j]
            ans.append(0 if x == -1 or y == -1 or x >= y else pre_sum[y] - pre_sum[x])
        return ans

    # https://leetcode-cn.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/ 满足条件的子序列数目
    def numSubseq(self, nums: List[int], target: int) -> int:
        mod = int(1e9 + 7)
        temp_nums = sorted(nums)
        n = len(temp_nums)
        res, l, r = 0, 0, n-1
        while l <= r:
            if temp_nums[l] + temp_nums[r] > target:
                r -= 1
            else:
                res = (res + self.quick_pow(2, r-l, mod)) % mod
                l += 1
        return int(res % mod)

    def quick_pow(self, a: int, n: int, mod=1e9):
        res = 1
        while n:
            if n & 1:
                res = (res * a) % mod
            a = (a % mod)**2 % mod
            n >>= 1
        return res

    # https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/ N 叉树的前序遍历
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        ans = []
        stack = [root]
        while stack:
            temp = stack.pop()
            ans.append(temp.val)
            stack.extend(reversed(temp.children))
        return ans

    # https://leetcode-cn.com/problems/count-nodes-with-the-highest-score/ 统计最高分的节点数目
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        if not parents:
            return []
        n = len(parents)
        cnt, max_val = 0, 0
        children = [[] for _ in range(n)]
        for node, parent in enumerate(parents):
            if node:
                children[parent].append(node)

        def dfs(nod: int) -> int:
            curr_val, total = 1, 1
            nonlocal max_val, cnt
            for child in children[nod]:
                val = dfs(child)
                curr_val *= val
                total += val
            if nod:
                curr_val *= (n - total)
            if curr_val == max_val:
                cnt += 1
            elif curr_val > max_val:
                max_val, cnt = curr_val, 1
            return total
        dfs(0)
        return cnt

    # https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal/ N 叉树的后序遍历
    def postorder(self, root: 'Node') -> List[int]:
        ans = []
        if not root:
            return []
        stack = [root]
        while stack:
            temp = stack.pop()
            ans.append(temp.val)
            stack.extend(temp.children)
        return reversed(ans)

    # https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/ 两个列表的最小索引总和
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        if not (list1 and list2):
            return []
        maps = {s: i for i, s in enumerate(list1)}
        idx_sum = inf
        ans = []
        for i, s in enumerate(list2):
            if s not in maps:
                continue
            curr_sum = i + maps[s]
            if curr_sum < idx_sum:
                idx_sum = curr_sum
                ans = [s]
            elif curr_sum == idx_sum:
                ans.append(s)
        return ans

    # https://leetcode-cn.com/problems/count-number-of-maximum-bitwise-or-subsets/ 统计按位或能得到最大值的子集数目
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        max_val, cnt = 0, 0
        for i in range(1, 1 << n):
            curr_val = reduce(or_, (num for j, num in enumerate(nums) if (i >> j) & 1), 0)
            if max_val < curr_val:
                max_val, cnt = curr_val, 1
            elif max_val == curr_val:
                cnt += 1
        return cnt

    # https://leetcode-cn.com/problems/longest-word-in-dictionary/ 词典中最长的单词
    def longestWord(self, words: List[str]) -> str:
        if not words:
            return ''
        root = Trie()
        for word in words:
            root.insert(word)
        ans = ''
        for word in words:
            s, t = len(word), len(ans)
            if root.search(word) and (s > t or (s == t and word < ans)):
                ans = word
        return ans

    # https://leetcode-cn.com/problems/construct-string-from-binary-tree/ 根据二叉树创建字符串
    def tree2str(self, root: Optional[TreeNode]) -> str:
        if not root:
            return ""
        if not root.left and not root.right:
            return str(root.val)
        if not root.right:
            return f"{root.val}({self.tree2str(root.left)})"
        return f"{root.val}({self.tree2str(root.left)})({self.tree2str(root.right)})"

    # https://leetcode-cn.com/problems/two-sum-iv-input-is-a-bst/ 两数之和 IV - 输入 BST
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        if not root:
            return False
        left, right = [root], [root]
        l, r = root, root
        while l.left:
            l = l.left
            left.append(l)
        while r.right:
            r = r.right
            right.append(r)
        while l != r:
            sum_val = l.val + r.val
            if sum_val == k:
                return True
            elif sum_val < k:
                left.pop()
                node = l.right
                while node:
                    left.append(node)
                    node = node.left
                l = left[-1]
            else:
                right.pop()
                node = r.left
                while node:
                    right.append(node)
                    node = node.right
                r = right[-1]
        return False

    # https://leetcode-cn.com/problems/remove-colored-pieces-if-both-neighbors-are-the-same-color/ 如果相邻两个颜色均相同则删除当前颜色
    def winnerOfGame(self, colors: str) -> bool:
        if not colors:
            return False
        # 统计碎片
        ans = [0] * 2
        c, cnt = '', 1
        for s in colors:
            if s != c:
                c, cnt = s, 1
            else:
                cnt += 1
                if cnt >= 3:
                    ans[ord(s) - ord('A')] += 1
        return ans[0] > ans[1]

    # https://leetcode-cn.com/problems/k-th-smallest-in-lexicographical-order/ 字典序的第K小数字
    def findKthNumber(self, n: int, k: int) -> int:
        idx, prefix = 1, 1
        while idx < k:
            cnt = self.get_cnt(prefix, n)
            curr_idx = idx + cnt
            if curr_idx <= k:
                prefix += 1
                idx = curr_idx
            else:
                prefix *= 10
                idx += 1
        return prefix

    def get_cnt(self, prefix: int, n: int) -> int:
        cnt, l, r = 0, prefix, prefix + 1
        while l <= n:
            real_right = min(r, n + 1)
            cnt += real_right - l
            l *= 10
            r *= 10
        return cnt

    # https://leetcode-cn.com/problems/image-smoother/ 图片平滑器
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        m, n = len(img), len(img[0])
        ans, prefix_sum = [[0] * n for _ in range(m)], [[0] * (n + 1) for _ in range(m + 1)]
        for i, j in product(range(1, m + 1), range(1, n + 1)):
            prefix_sum[i][j] = prefix_sum[i][j-1] + prefix_sum[i-1][j] - prefix_sum[i-1][j-1] + img[i-1][j-1]
        for i, j in product(range(m), range(n)):
            a, b = max(0, i - 1), max(0, j - 1)
            c, d = min(i + 1, m - 1), min(j + 1, n - 1)
            cnt = (c - a + 1) * (d - b + 1)
            total = prefix_sum[c+1][d+1] - prefix_sum[a][d+1] - prefix_sum[c+1][b] + prefix_sum[a][b]
            ans[i][j] = total // cnt
        return ans

    # https://leetcode-cn.com/problems/factorial-trailing-zeroes/ 阶乘后的零
    def trailingZeroes(self, n: int) -> int:
        ans = 0
        while n >= 5:
            ans += n // 5
            n //= 5
        return ans

    # https://leetcode-cn.com/problems/baseball-game/  棒球比赛
    def calPoints(self, ops: List[str]) -> int:
        if not ops:
            return 0
        cache, ans = [], 0
        for op in ops:
            if op == 'C':
                ans -= cache.pop(-1)
            elif op == 'D':
                d = cache[-1] * 2
                cache.append(d)
                ans += d
            elif op == '+':
                p = cache[-1] + cache[-2]
                cache.append(p)
                ans += p
            else:
                val = int(op)
                cache.append(val)
                ans += val
        return ans

    # https://leetcode-cn.com/problems/find-missing-observations/ 找出缺失的观测数据
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        if not rolls:
            return []
        curr_sum, m = sum(rolls), len(rolls)
        miss_sum = (m + n) * mean - curr_sum
        if not (n <= miss_sum <= 6 * n):
            return []
        val, rest = divmod(miss_sum, n)
        return [val + 1] * rest + [val] * (n - rest)

    # https://leetcode-cn.com/problems/binary-number-with-alternating-bits/ 交替位二进制数
    def hasAlternatingBits(self, n: int) -> bool:
        a = n ^ (n >> 1)
        return (a & (a + 1)) == 0

    # https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/ 考试的最大困扰度
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        if not answerKey:
            return 0

        def find_len(s: str) -> int:
            i, j, v, n = 0, 0, 0, len(answerKey)
            while i < n:
                if answerKey[i] == s:
                    v += 1
                if v > k:
                    if answerKey[j] == s:
                        v -= 1
                    j += 1
                i += 1
            return i - j
        return max(find_len('T'), find_len('F'))

    # https://leetcode-cn.com/problems/find-servers-that-handled-most-number-of-requests/ 找到处理最多请求的服务器
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        cnts = [0] * k
        busy = []
        free = SortedList(range(k))
        max_cnt = 0
        for i in range(len(arrival)):
            start = arrival[i]
            end = start + load[i]
            while busy and busy[0][0] <= start:
                free.add(busy[0][1])
                heappop(busy)
            if (idx := free.bisect_left(i % k)) == len(free) == (idx := free.bisect_left(0)):
                continue
            work = free.pop(idx)
            heappush(busy, (end, work))
            cnts[work] += 1
            max_cnt = max(max_cnt, cnts[work])
        return [j for j in range(k) if cnts[j] == max_cnt]

    # https://leetcode-cn.com/problems/self-dividing-numbers/ 自除数
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        ans = []
        if left > right:
            return ans
        for i in range(left, right + 1):
            temp, flag = i, True
            while temp:
                temp, n = divmod(temp, 10)
                if not (n and i % n == 0):
                    flag = False
                    break
            if flag:
                ans.append(i)
        return ans

    # https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/ 寻找比目标字母大的最小字母
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        if not letters:
            return ""
        if ord(letters[-1]) <= ord(target):
            return letters[0]
        l, r = 0, len(letters) - 1
        while l < r:
            mid = (l + r) >> 1
            if ord(letters[mid]) <= ord(target):
                l = mid + 1
            else:
                r = mid
        return letters[r]

    # https://leetcode-cn.com/problems/array-of-doubled-pairs/ 二倍数对数组
    def canReorderDoubled(self, arr: List[int]) -> bool:
        if not arr:
            return False
        num_map = collections.Counter(arr)
        for num in sorted(num_map, key=abs):
            if num_map[num] > num_map[2 * num]:
                return False
            num_map[2 * num] -= num_map[num]
        return True

    # https://leetcode-cn.com/problems/prime-number-of-set-bits-in-binary-representation/ 二进制表示中质数个计算置位
    def countPrimeSetBits(self, left: int, right: int) -> int:
        return sum(self.isprime(i.bit_count()) for i in range(left, right + 1))

    def isprime(self, x: int) -> bool:
        if x < 2:
            return False
        i = 2
        while i * i <= x:
            if x % i == 0:
                return False
            i += 1
        return True


# https://leetcode-cn.com/problems/range-sum-query-mutable/ 区域和检索 - 数组可修改
class NumArray:

    def __lowbit(self, x: int) -> int:
        return x & -x

    def __add(self, x: int, v: int):
        while x <= self.n:
            self.tree[x] += v
            x += self.__lowbit(x)

    def __query(self, x: int) -> int:
        ans = 0
        while x:
            ans += self.tree[x]
            x -= self.__lowbit(x)
        return ans

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.n = len(nums)
        self.tree = [0] * (self.n + 1)
        for i, num in enumerate(nums, 1):
            self.__add(i, num)

    def update(self, index: int, val: int) -> None:
        self.__add(index + 1, val - self.nums[index])
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        return self.__query(right + 1) - self.__query(left)


# 字典树
class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

    def insert(self, s: str):
        node = self
        for ch in s:
            idx = ord(ch) - ord('a')
            if not node.children[idx]:
                node.children[idx] = Trie()
            node = node.children[idx]
        node.is_end = True

    def search(self, s: str) -> bool:
        node = self
        for ch in s:
            idx = ord(ch) - ord('a')
            # node.children[idx].is_end 单词逐步添加一个字母组成
            if not (node.children[idx] and node.children[idx].is_end):
                return False
            node = node.children[idx]
        return True


# https://leetcode-cn.com/problems/all-oone-data-structure/ 全 O(1) 的数据结构
class LfuNode:
    def __init__(self, key='', cnt=1):
        self.keys = {key}
        self.cnt = cnt
        self.pre = None
        self.nxt = None

    def remove(self):
        self.pre.nxt = self.nxt
        self.nxt.pre = self.pre

    def insert(self, node: 'LfuNode') -> 'LfuNode':
        node.nxt = self.nxt
        self.nxt.pre = node
        self.nxt = node
        node.pre = self
        return node

    
class AllOne:

    def __init__(self):
        self.root = LfuNode()
        self.nodes = {}
        self.root.pre = self.root
        self.root.nxt = self.root

    def inc(self, key: str) -> None:
        if key in self.nodes:
            lfu_node = self.nodes[key]
            nxt = lfu_node.nxt
            cnt = lfu_node.cnt
            lfu_node.keys.discard(key)
            if nxt is not self.root and nxt.cnt == cnt + 1:
                nxt.keys.add(key)
                self.nodes[key] = nxt
            else:
                self.nodes[key] = lfu_node.insert(LfuNode(key, cnt + 1))
            if not lfu_node.keys:
                lfu_node.remove()
        else:
            nxt = self.root.nxt
            if nxt is not self.root and nxt.cnt == 1:
                nxt.keys.add(key)
                self.nodes[key] = nxt
            else:
                self.nodes[key] = self.root.insert(LfuNode(key))

    def dec(self, key: str) -> None:
        if key not in self.nodes:
            return
        lfu_node = self.nodes[key]
        pre = lfu_node.pre
        cnt = lfu_node.cnt
        keys = lfu_node.keys
        keys.discard(key)
        if cnt == 1:
            del self.nodes[key]
        elif pre is not self.root and pre.cnt == cnt - 1:
            pre.keys.add(key)
            self.nodes[key] = pre
        else:
            self.nodes[key] = pre.insert(LfuNode(key, cnt - 1))
        if not keys:
            lfu_node.remove()

    def getMaxKey(self) -> str:
        return "" if self.root.pre is self.root else next(iter(self.root.pre.keys))

    def getMinKey(self) -> str:
        return "" if self.root.nxt is self.root else next(iter(self.root.nxt.keys))


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


# https://leetcode-cn.com/problems/simple-bank-system/ 简易银行系统
class Bank:

    def __init__(self, balance: List[int]):
        self.balance = balance
        self.cnt = len(balance)

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if self.is_illegal_account(account1) or self.is_illegal_account(account2) or \
                self.balance[account1 - 1] < money:
            return False
        self.balance[account1 - 1] -= money
        self.balance[account2 - 1] += money
        return True

    def deposit(self, account: int, money: int) -> bool:
        if self.is_illegal_account(account):
            return False
        self.balance[account - 1] += money
        return True

    def withdraw(self, account: int, money: int) -> bool:
        if self.is_illegal_account(account) or self.balance[account - 1] < money:
            return False
        self.balance[account - 1] -= money
        return True

    def is_illegal_account(self, account: int) -> bool:
        return account > self.cnt or account <= 0


if __name__ == '__main__':
    s = Solution()
    c = collections.Counter("aaaaabbbcc")
    print(c)
    print(type(c))
    print(sorted(c))
