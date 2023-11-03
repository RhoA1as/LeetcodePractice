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
