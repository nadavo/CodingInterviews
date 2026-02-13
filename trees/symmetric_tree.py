from typing import List


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def preorder_DFS_recursive(self, node: TreeNode, tree: List[int]) -> List[int]:
        if node:
            tree.append(node.val)
            self.preorder_DFS_recursive(node.left, tree)
            self.preorder_DFS_recursive(node.right, tree)
        else:
            tree.append(None)
        return tree

    def preorder_DFS_iterative(self, node: TreeNode) -> List[int]:
        visited, stack = list(), list()
        current = node
        while True:
            if not current and not stack:
                return visited
            while current:
                visited.append(current.val)
                stack.append(current.right)
                current = current.left
            else:
                visited.append(None)
            if stack:
                current = stack.pop()

    def reverse_preorder_DFS_recursive(
        self, node: TreeNode, tree: List[int]
    ) -> List[int]:
        if node:
            tree.append(node.val)
            self.reverse_preorder_DFS_recursive(node.right, tree)
            self.reverse_preorder_DFS_recursive(node.left, tree)
        else:
            tree.append(None)
        return tree

    def reverse_preorder_DFS_iterative(self, node: TreeNode) -> List[int]:
        visited, stack = list(), list()
        current = node
        while True:
            if not current and not stack:
                return visited
            while current:
                visited.append(current.val)
                stack.append(current.left)
                current = current.right
            else:
                visited.append(None)
            if stack:
                current = stack.pop()

    early_exit: bool = False

    def reverse_preorder_DFS_recursive_bool(
        self, node: TreeNode, tree: List[int]
    ) -> bool:
        if self.early_exit:
            return False

        if node:
            if node.val == tree[0]:
                tree.pop(0)
                return self.reverse_preorder_DFS_recursive_bool(
                    node.right, tree
                ) and self.reverse_preorder_DFS_recursive_bool(node.left, tree)
            else:
                self.early_exit = True
                return False
        else:
            if tree[0] is None:
                tree.pop(0)
                return True
            else:
                self.early_exit = True
                return False

    def reverse_preorder_DFS_iterative_bool(
        self, node: TreeNode, tree: List[int]
    ) -> bool:
        visited, stack = list(), list()
        current = node
        while True:
            if not current and not stack:
                return visited == tree
            while current:
                visited.append(current.val)
                stack.append(current.left)
                current = current.right
            else:
                visited.append(None)
            if stack:
                current = stack.pop()

    def reverse_preorder_DFS_iterative_bool_express(
        self, node: TreeNode, tree: List[int]
    ) -> bool:
        i = 0
        stack = list()
        current = node
        while current or stack:
            while current:
                if i >= len(tree) or current.val != tree[i]:
                    return False
                stack.append(current.left)
                current = current.right
                i += 1
            if stack:
                current = stack.pop()
                i += 1
        return i == len(tree)

    def symmetric_BFS_iterative_bool_express(
        self, left_root: TreeNode, right_root: TreeNode
    ) -> bool:
        l = left_root
        r = right_root
        stack = list()
        while l or r or stack:
            while l or r:
                if self.get_val(l) != self.get_val(r):
                    return False
                stack.append((l.right, r.left))
                l = l.left
                r = r.right
            if stack:
                l, r = stack.pop()

        return True

    @staticmethod
    def get_val(node: TreeNode) -> int:
        return node.val if node else None

    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return False

        if not root.left and not root.right:
            return True
        elif self.get_val(root.left) != self.get_val(root.right):
            return False

        #### RECURSIVE
        #         left_tree = list()
        #         self.preorder_DFS_recursive(root.left, left_tree)
        #         # right_tree = list()
        #         # self.reverse_preorder_DFS(root.right, right_tree)

        #         # return left_tree == right_tree
        #         return self.reverse_preorder_DFS_bool_recursive(root.right, left_tree)

        #### ITERATIVE
        # left_tree = self.preorder_DFS_iterative(root.left)
        # return self.reverse_preorder_DFS_iterative_bool_express(root.right, left_tree)

        #### Parallel
        return self.symmetric_BFS_iterative_bool_express(root.left, root.right)
