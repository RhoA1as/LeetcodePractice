# leetcode hot100 https://leetcode.cn/studyplan/top-100-liked/
import bisect
import collections
import heapq
from functools import reduce
from math import inf
from typing import List, Optional


# https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
def twoSum(self, nums: List[int], target: int) -> List[int]:
    m = {}
    for i, v in enumerate(nums):
        if target - v in m:
            return [m[target - v], i]
        else:
            m[v] = i
    return []


# https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    m = collections.defaultdict(list)
    for s in strs:
        m["".join(sorted(s))].append(s)
    return list(m.values())


# https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked
def longestConsecutive(self, nums: List[int]) -> int:
    ns, res = set(nums), 0
    for i in ns:
        if i - 1 not in ns:
            cur = 1
            while i + cur in ns:
                cur += 1
            res = max(res, cur)
    return res


# https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked
def moveZeroes(self, nums: List[int]) -> None:
    l = r = 0
    s = len(nums)
    while r < s:
        if nums[r]:
            if l != r:
                nums[l], nums[r] = nums[r], nums[l]
            l += 1
        r += 1


# https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked
def maxArea(self, height: List[int]) -> int:
    l, r, res = 0, len(height) - 1, 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        res = max(area, res)
        if height[l] <= height[r]:
            l += 1
        else:
            r -= 1
    return res


# https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked
def threeSum(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    nums.sort()
    ans = []
    for i in range(n - 2):
        if i and nums[i] == nums[i - 1]:
            continue
        if nums[i] + nums[i + 1] + nums[i + 2] > 0:
            break
        if nums[i] + nums[n - 2] + nums[n - 1] < 0:
            continue
        j, k = i + 1, n - 1
        while j < k:
            if j > i + 1 and nums[j] == nums[j - 1]:
                j += 1
                continue
            while j < k and nums[j] + nums[k] + nums[i] > 0:
                k -= 1
            if j == k:
                break
            res = nums[j] + nums[k] + nums[i]
            if res == 0:
                ans.append([nums[i], nums[j], nums[k]])
                k -= 1
            j += 1
    return ans


# https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked
def lengthOfLongestSubstring(self, s: str) -> int:
    i = res = 0
    st = set()
    for j, v in enumerate(s):
        while v in st:
            st.remove(s[i])
            i += 1
        st.add(v)
        res = max(res, j - i + 1)
    return res


# https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked
def findAnagrams(self, s: str, p: str) -> List[int]:
    n, l = len(p), len(s)
    if n > l:
        return []
    if n == l and s == p:
        return [0]
    s_cnt = [0] * 26
    p_cnt = [0] * 26
    ans = []
    for i in range(n):
        s_cnt[ord(s[i]) - ord("a")] += 1
        p_cnt[ord(p[i]) - ord("a")] += 1
    if s_cnt == p_cnt:
        ans.append(0)
    for i in range(l - n):
        s_cnt[ord(s[i]) - ord("a")] -= 1
        s_cnt[ord(s[i + n]) - ord("a")] += 1
        if s_cnt == p_cnt:
            ans.append(i + 1)
    return ans


# https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked
def subarraySum(self, nums: List[int], k: int) -> int:
    s, m, ans = 0, collections.defaultdict(int), 0
    m[0] = 1
    for num in nums:
        s += num
        ans += m[s - k]
        m[s] += 1
    return ans


# https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked
def maxSubArray(self, nums: List[int]) -> int:
    cur, res = 0, nums[0]
    for num in nums:
        cur = max(num, cur + num)
        res = max(res, cur)
    return res


# https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    res = []
    intervals.sort(key=lambda x: x[0])
    for l in intervals:
        if not res or res[-1][1] < l[0]:
            res.append(l)
        else:
            res[-1][1] = max(res[-1][1], l[1])
    return res


# https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked
def rotate(self, nums: List[int], k: int) -> None:
    k %= len(nums)
    nums.reverse()
    nums[:k] = nums[:k][::-1]
    nums[k:] = nums[k:][::-1]


# https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=top-100-liked
def productExceptSelf(self, nums: List[int]) -> List[int]:
    n = len(nums)
    res = [0] * n
    res[0] = 1
    for i in range(1, n):
        res[i] = res[i - 1] * nums[i - 1]
    r = 1
    for i in range(n - 2, -1, -1):
        r *= nums[i + 1]
        res[i] *= r
    return res


# https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked
def setZeroes(self, matrix: List[List[int]]) -> None:
    row, col = set(), set()
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        for j in range(n):
            if not matrix[i][j]:
                row.add(i)
                col.add(j)

    for i in range(m):
        for j in range(n):
            if i in row or j in col:
                matrix[i][j] = 0


# https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    l, r, t, b = 0, len(matrix[0]) - 1, 0, len(matrix) - 1
    size = (r + 1) * (b + 1)
    res = []
    while size >= 1:
        for i in range(l, r + 1):
            res.append(matrix[t][i])
            size -= 1
        if not size:
            break
        t += 1
        for i in range(t, b + 1):
            res.append(matrix[i][r])
            size -= 1
        if not size:
            break
        r -= 1
        for i in range(r, l - 1, -1):
            res.append(matrix[b][i])
            size -= 1
        if not size:
            break
        b -= 1
        for i in range(b, t - 1, -1):
            res.append(matrix[i][l])
            size -= 1
        l += 1
    return res


# https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-100-liked
def rotate(self, matrix: List[List[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(m):
        for j in range(n // 2):
            matrix[i][j], matrix[i][n - 1 - j] = matrix[i][n - 1 - j], matrix[i][j]


class ListNode:
    def __init__(self, x=0):
        self.val = x
        self.next = None


# https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    if not (headA and headB):
        return None
    pa, pb = headA, headB
    while pa is not pb:
        pa = pa.next if pa else headB
        pb = pb.next if pb else headA
    return pa


# https://leetcode.cn/problems/reverse-linked-list/description/?envType=study-plan-v2&envId=top-100-liked
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return head
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


# https://leetcode.cn/problems/palindrome-linked-list/description/?envType=study-plan-v2&envId=top-100-liked
def isPalindrome(self, head: Optional[ListNode]) -> bool:
    def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        prev, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev

    if not head:
        return True
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    head1 = reverseList(slow)
    while head1:
        if head1.val != head.val:
            return False
        head1 = head1.next
        head = head.next
    return True


# https://leetcode.cn/problems/linked-list-cycle/?envType=study-plan-v2&envId=top-100-liked
def hasCycle(self, head: Optional[ListNode]) -> bool:
    if not head:
        return False
    f = s = head
    while f and f.next:
        f = f.next.next
        s = s.next
        if f is s:
            return True
    return False


# https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2&envId=top-100-liked
def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
    def hasCycle(head: Optional[ListNode]) -> Optional[ListNode]:
        f = s = head
        while f and f.next:
            f = f.next.next
            s = s.next
            if f is s:
                return f
        return None

    if not head:
        return None
    f = hasCycle(head)
    if not f:
        return None
    h = head
    while h is not f:
        h = h.next
        f = f.next
    return h


# https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=study-plan-v2&envId=top-100-liked
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    if not (list1 and list2):
        return list1 if list1 else list2
    temp = ListNode()
    l1, l2, t = list1, list2, temp
    while l1 and l2:
        val1, val2 = l1.val, l2.val
        if val1 < val2:
            t.next = l1
            l1 = l1.next
        else:
            t.next = l2
            l2 = l2.next
        t = t.next
    if l1 or l2:
        t.next = l1 if l1 else l2
    return temp.next


# https://leetcode.cn/problems/add-two-numbers/description/?envType=study-plan-v2&envId=top-100-liked
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    head = ListNode()
    h1, h2, t = l1, l2, head
    ext = 0
    while l1 or l2:
        val1 = val2 = 0
        if l1:
            val1 = l1.val
            l1 = l1.next
        if l2:
            val2 = l2.val
            l2 = l2.next
        s = val1 + val2 + ext
        ext = s // 10
        t.next = ListNode(s % 10)
        t = t.next
    if ext:
        t.next = ListNode(1)
    return head.next


# https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2&envId=top-100-liked
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    i, j = 0, n - 1
    while i < m and j >= 0:
        if matrix[i][j] == target:
            return True
        if matrix[i][j] < target:
            i += 1
        else:
            j -= 1
    return False


# https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&envId=top-100-liked
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    if not head:
        return None
    t = ListNode()
    t.next = head
    t1 = t2 = t
    for _ in range(n):
        if not t1:
            return None
        t1 = t1.next
    while t1.next:
        t1 = t1.next
        t2 = t2.next
    t2.next = t2.next.next
    return t.next


# https://leetcode.cn/problems/swap-nodes-in-pairs/description/?envType=study-plan-v2&envId=top-100-liked
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return head
    f = t = ListNode()
    t.next = head
    while f.next and f.next.next:
        n1, n2 = f.next, f.next.next
        f.next = n2
        n1.next = n2.next
        n2.next = n1
        f = n1
    return t.next


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


# https://leetcode.cn/problems/copy-list-with-random-pointer/description/?envType=study-plan-v2&envId=top-100-liked
def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
    if not head:
        return head
    cache = {}
    h = hf = Node(0)
    t = head
    while t:
        temp = Node(t.val)
        hf.next = temp
        cache[t] = temp
        t = t.next
        hf = hf.next
    hf, t = h.next, head
    while t:
        if t.random:
            hf.random = cache[t.random]
        hf = hf.next
        t = t.next
    return h.next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# https://leetcode.cn/problems/binary-tree-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    res, st = [], []
    while root or st:
        while root:
            st.append(root)
            root = root.left
        root = st.pop(-1)
        res.append(root.val)
        root = root.right
    return res


# https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


# https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    root.left, root.right = root.right, root.left
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root


# https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2&envId=top-100-liked
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def check(root0: Optional[TreeNode], root1: Optional[TreeNode]):
        if not (root0 or root1):
            return True
        if not (root0 and root1):
            return False
        return root0.val == root1.val and check(root0.left, root1.right) and check(root0.right, root1.left)

    return check(root, root)


# https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked
def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    self.res = 0

    def depth(r: Optional[TreeNode]) -> int:
        if not r:
            return 0
        left = depth(r.left)
        right = depth(r.right)
        self.res = max(self.res, left + right)
        return 1 + max(left, right)

    depth(root)
    return self.res


# https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=study-plan-v2&envId=top-100-liked
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    queue = [root]
    res = []
    while queue:
        size = len(queue)
        t = []
        for _ in range(size):
            node = queue.pop(0)
            t.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(t)
    return res


# https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked
def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    if not nums:
        return None

    def translate(left, right) -> Optional[TreeNode]:
        if left > right:
            return None
        mid = left + (right - left) // 2
        root = TreeNode(nums[mid])
        root.left = translate(left, mid - 1)
        root.right = translate(mid + 1, right)
        return root

    return translate(0, len(nums) - 1)


# https://leetcode.cn/problems/validate-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True
    inorder = -inf
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop(-1)
        if root.val <= inorder:
            return False
        inorder = root.val
        root = root.right
    return True


# https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2&envId=top-100-liked
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop(-1)
        k -= 1
        if not k:
            return root.val
        root = root.right
    return -1


# https://leetcode.cn/problems/binary-tree-right-side-view/description/?envType=study-plan-v2&envId=top-100-liked
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    queue = [root]
    res = []
    while queue:
        size = len(queue)
        for i in range(size):
            t = queue.pop(0)
            if i == size - 1:
                res.append(t.val)
            if t.left:
                queue.append(t.left)
            if t.right:
                queue.append(t.right)
    return res


# https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2&envId=top-100-liked
def flatten(self, root: Optional[TreeNode]) -> None:
    if not root:
        return
    t = root
    while t:
        if t.left:
            f = t.left
            while f.right:
                f = f.right
            f.right = t.right
            t.right = t.left
            t.left = None
        t = t.right


# https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?envType=study-plan-v2&envId=top-100-liked
def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    if not (preorder and inorder):
        return None
    if len(preorder) == 1:
        return TreeNode(preorder[0])
    val = preorder[0]
    root = TreeNode(val)
    idx = 0
    for i, v in enumerate(inorder):
        if v == val:
            idx = i
            break
    root.left = self.buildTree(preorder[1:idx + 1], inorder[:idx])
    root.right = self.buildTree(preorder[idx + 1:], inorder[idx + 1:])
    return root


# https://leetcode.cn/problems/path-sum-iii/description/?envType=study-plan-v2&envId=top-100-liked
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
    cache = collections.defaultdict(int)
    cache[0] = 1

    def dfs(node: Optional[TreeNode], s: int) -> int:
        if not node:
            return 0
        res = 0
        s += node.val
        res += cache[s - targetSum]
        cache[s] += 1
        res += dfs(node.left, s)
        res += dfs(node.right, s)
        cache[s] -= 1
        return res

    return dfs(root, 0)


# https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root or root is p or root is q:
        return root
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    if not left:
        return right
    if not right:
        return left
    return root


# https://leetcode.cn/problems/single-number/?envType=study-plan-v2&envId=top-100-liked
def singleNumber(self, nums: List[int]) -> int:
    return reduce(lambda x, y: x ^ y, nums)


# https://leetcode.cn/problems/majority-element/description/?envType=study-plan-v2&envId=top-100-liked
def majorityElement(self, nums: List[int]) -> int:
    nums.sort()
    return nums[len(nums) // 2]


# https://leetcode.cn/problems/sort-colors/description/?envType=study-plan-v2&envId=top-100-liked
def sortColors(self, nums: List[int]) -> None:
    n = len(nums)
    p0 = p1 = 0
    for i in range(n):
        if nums[i] == 1:
            nums[p1], nums[i] = nums[i], nums[p1]
            p1 += 1
        elif not nums[i]:
            nums[p0], nums[i] = nums[i], nums[p0]
            if p0 < p1:
                nums[p1], nums[i] = nums[i], nums[p1]
            p0 += 1
            p1 += 1


# https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=top-100-liked
def maxProfit(self, prices: List[int]) -> int:
    cost, res = inf, 0
    for p in prices:
        cost = min(cost, p)
        res = max(res, p - cost)
    return res


# https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2&envId=top-100-liked
def climbStairs(self, n: int) -> int:
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


# https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked
def isValid(self, s: str) -> bool:
    m = {'(': 1, ')': -1, '[': 2, ']': -2, '{': 3, '}': -3}
    n = []
    if not s:
        return True
    for c in s:
        if m[c] < 0:
            if len(n) == 0 or n[-1] + m[c] != 0:
                return False
            n.pop(-1)
        else:
            n.append(m[c])
    return len(n) == 0


# https://leetcode.cn/problems/find-the-duplicate-number/description/?envType=study-plan-v2&envId=top-100-liked
def findDuplicate(self, nums: List[int]) -> int:
    fast, slow = nums[nums[0]], nums[0]
    while fast != slow:
        slow = nums[slow]
        fast = nums[nums[fast]]
    slow = 0
    while fast != slow:
        slow = nums[slow]
        fast = nums[fast]
    return slow


# https://leetcode.cn/problems/daily-temperatures/description/?envType=study-plan-v2&envId=top-100-liked
def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    if not temperatures:
        return []
    n = len(temperatures)
    ans = [0] * n
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and temperatures[stack[-1]] <= temperatures[i]:
            stack.pop(-1)
        if stack:
            ans[i] = stack[-1] - i
        stack.append(i)
    return ans


# https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    i, j = 0, m * n - 1
    while i <= j:
        k = (j - i) // 2 + i
        v = matrix[k // n][k % n]
        if v < target:
            i = k + 1
        elif v > target:
            j = k - 1
        else:
            return True
    return False


# https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-100-liked
def searchInsert(self, nums: List[int], target: int) -> int:
    if not nums:
        return 0
    l, ans = 0, len(nums)
    r = ans - 1
    if target <= nums[0]:
        return 0
    if target > nums[r]:
        return ans
    while l <= r:
        m = l + (r - l) // 2
        if nums[m] == target:
            ans = m
            break
        if nums[m] > target:
            ans = m
            r = m - 1
        else:
            l = m + 1
    return ans


# https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
def searchRange(self, nums: List[int], target: int) -> List[int]:
    if not nums:
        return [-1, -1]

    def binary_search(low: bool):
        l, r = 0, len(nums) - 1
        ans = -1
        while l <= r:
            m = l + (r - l) >> 1
            mr = nums[m]
            if mr < target:
                l = m + 1
            elif mr > target:
                r = m - 1
            else:
                ans = m
                if low:
                    if m != 0 and nums[m - 1] == nums[m]:
                        r = m - 1
                    else:
                        break
                else:
                    if m != len(nums) - 1 and nums[m + 1] == nums[m]:
                        l = m + 1
                    else:
                        break
        return ans

    left = binary_search(True)
    return [-1, -1] if left == -1 else [left, binary_search(False)]


# https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
def search(self, nums: List[int], target: int) -> int:
    if not nums:
        return -1
    exist_left = target >= nums[0]
    n = len(nums)
    l, r = 0, n - 1
    while l <= r:
        m = l + (r - l) // 2
        if nums[m] == target:
            return m
        if exist_left:
            if nums[m] < nums[0] or nums[m] > target:
                r = m - 1
            else:
                l = m + 1
        else:
            if nums[m] > nums[n - 1] or nums[m] < target:
                l = m + 1
            else:
                r = m - 1
    return -1


# https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
def findMin(self, nums: List[int]) -> int:
    if not nums:
        return nums[0]
    l, r = 0, len(nums) - 1
    while l < r:
        m = l + (r - l) // 2
        if nums[m] > nums[r]:
            l = m + 1
        else:
            r = m
    return nums[l]


# https://leetcode.cn/problems/permutations/description/?envType=study-plan-v2&envId=top-100-liked
def permute(self, nums: List[int]) -> List[List[int]]:
    if not nums:
        return []
    res = []
    n = len(nums)

    def backtrack(idx: int):
        if idx == n - 1:
            res.append(nums[:])
            return
        for i in range(idx, n):
            nums[idx], nums[i] = nums[i], nums[idx]
            backtrack(idx + 1)
            nums[idx], nums[i] = nums[i], nums[idx]

    backtrack(0)
    return res


# https://leetcode.cn/problems/subsets/description/?envType=study-plan-v2&envId=top-100-liked
def subsets(self, nums: List[int]) -> List[List[int]]:
    if not nums:
        return [[]]
    ans = [[]]
    for i in nums:
        t = []
        for li in ans:
            t.append(li + [i])
        ans += t
    return ans


# https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/?envType=study-plan-v2&envId=top-100-liked
def letterCombinations(self, digits: str) -> List[str]:
    if not digits:
        return []
    n = len(digits)
    ans = []
    t = []
    m = {'2': 'abc',
         '3': 'def',
         '4': 'ghi',
         '5': 'jkl',
         '6': 'mno',
         '7': 'pqrs',
         '8': 'tuv',
         '9': 'wxyz'}

    def backtrack(idx: int):
        if idx == n:
            ans.append("".join(t))
            return
        s = m[digits[idx]]
        for c in s:
            t.append(c)
            backtrack(idx + 1)
            t.pop()

    backtrack(0)
    return ans


# https://leetcode.cn/problems/combination-sum/description/?envType=study-plan-v2&envId=top-100-liked
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    if not candidates:
        return []
    candidates.sort()
    ans = []
    t = []
    n = len(candidates)

    def backtrack(idx: int, left: int):
        if not left:
            ans.append(t[:])
            return
        for i in range(idx, n):
            if left < candidates[i]:
                break
            t.append(candidates[i])
            backtrack(i, left - candidates[i])
            t.pop()

    backtrack(0, target)
    return ans


# https://leetcode.cn/problems/jump-game/description/?envType=study-plan-v2&envId=top-100-liked
def canJump(self, nums: List[int]) -> bool:
    end, n = 0, len(nums)
    for i in range(n):
        if i <= end:
            end = max(end, i + nums[i])
            if end >= n - 1:
                return True
        else:
            break
    return False


# https://leetcode.cn/problems/jump-game-ii/description/?envType=study-plan-v2&envId=top-100-liked
def jump(self, nums: List[int]) -> int:
    end = step = right = 0
    n = len(nums)
    for i in range(n - 1):
        right = max(right, i + nums[i])
        if i == end:
            step += 1
            end = right
    return step


# https://leetcode.cn/problems/next-permutation/description/?envType=study-plan-v2&envId=top-100-liked
def nextPermutation(self, nums: List[int]) -> None:
    if not nums:
        return
    n, left = len(nums), -1
    for i in range(n - 2, -1, -1):
        if nums[i] < nums[i + 1]:
            left = i
            break
    a, b = left + 1, n - 1
    while a < b:
        nums[a], nums[b] = nums[b], nums[a]
        a += 1
        b -= 1
    if left == -1:
        return
    idx = bisect.bisect_right(nums[left + 1:], nums[left]) + left + 1
    nums[idx], nums[left] = nums[left], nums[idx]


# https://leetcode.cn/problems/first-missing-positive/?envType=study-plan-v2&envId=top-100-liked
def firstMissingPositive(self, nums: List[int]) -> int:
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


# https://leetcode.cn/problems/binary-tree-maximum-path-sum/?envType=study-plan-v2&envId=top-100-liked
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    ans = -inf

    def backtrack(node: Optional[TreeNode]):
        nonlocal ans
        if not node:
            return 0
        left_val = max(backtrack(node.left), 0)
        right_val = max(backtrack(node.right), 0)
        ans = max(ans, node.val + left_val + right_val)
        return node.val + max(left_val, right_val)

    backtrack(root)
    return int(ans)


# https://leetcode.cn/problems/top-k-frequent-elements/submissions/105611194/?envType=study-plan-v2&envId=top-100-liked
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    if not nums:
        return []
    cnt = collections.Counter(nums)
    return heapq.nlargest(k, cnt.keys(), key=lambda x: cnt[x])