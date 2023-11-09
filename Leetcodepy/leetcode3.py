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


# https://leetcode.cn/problems/find-the-longest-balanced-substring-of-a-binary-string/ 最长平衡子字符串
def findTheLongestBalancedSubstring(self, s: str) -> int:
    res = cnt0 = cnt1 = 0
    for i in range(len(s)):
        if s[i] == "1":
            cnt1 += 1
            res = max(res, 2 * min(cnt0, cnt1))
        elif i == 0 or s[i-1] == "1":
            cnt0 = 1
            cnt1 = 0
        else:
            cnt0 += 1
    return res