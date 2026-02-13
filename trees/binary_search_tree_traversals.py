from typing import List, Callable
from timeit import default_timer as timer
from collections import deque


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x: int, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right


def isValidBST(root: TreeNode) -> bool:
    return _isValidBST_recursive(root, None, None)


def _isValidBST_recursive(node: TreeNode, lower_bound=None, upper_bound=None) -> bool:
    if not node:
        return True
    if lower_bound is not None and node.val <= lower_bound:
        return False
    if upper_bound is not None and node.val >= upper_bound:
        return False
    return bool(
        _isValidBST_recursive(node.left, lower_bound, node.val)
        and _isValidBST_recursive(node.right, node.val, upper_bound)
    )


# Inorder Traversal (Left, Node, Right)
def inorder_DFS(node: TreeNode, result: List[int]) -> List[int]:
    if node:
        inorder_DFS(node.left, result)
        result.append(node.val)
        inorder_DFS(node.right, result)
        return result


def recursive_inorder_traversal(root: TreeNode) -> List[int]:
    visited = list()
    return inorder_DFS(root, visited)


def iterative_inorder_traversal(root: TreeNode) -> List[int]:
    visited = list()
    stack = list()
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        if stack:
            current = stack.pop()
            visited.append(current.val)
            current = current.right
    return visited


# Preorder Traversal (Node, Left, Right)
def preorder_DFS(node: TreeNode, result: List[int]) -> List[int]:
    if node:
        result.append(node.val)
        preorder_DFS(node.left, result)
        preorder_DFS(node.right, result)
        return result


def recursive_preorder_traversal(root: TreeNode) -> List[int]:
    visited = list()
    return preorder_DFS(root, visited)


def iterative_preorder_traversal(root: TreeNode) -> List[int]:
    visited, stack = list(), list()
    current = root
    while current or stack:
        while current:
            visited.append(current.val)
            if current.right:  # Depending if we allow unbalanced trees
                stack.append(current.right)
            current = current.left
        ## if we allow unbalanced trees
        # else:
        #     visited.append(None)
        if stack:
            current = stack.pop()
    return visited


def iterative_levelorder_BFS_traversal(root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    if not root.left and not root.right:
        return [[root.val]]

    queue = deque([root])
    tree = list()
    while queue:
        # Determine number of nodes in new level
        num_nodes_level = len(queue)
        level = list()
        # Only add the first num_nodes_level of nodes from queue
        while num_nodes_level:
            # BFS level order traversal
            current = queue.popleft()
            # Add current level node value to level
            level.append(current.val)
            num_nodes_level -= 1
            # Add next level nodes to queue
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        # Add level to tree
        tree.append(level)
    return tree


# Postorder Traversal (Left, Right, Node) or reversed(Node, Right, Left)
def postorder_DFS(node: TreeNode, result: List[int]) -> List[int]:
    if node:
        postorder_DFS(node.left, result)
        postorder_DFS(node.right, result)
        result.append(node.val)
        return result


def recursive_postorder_traversal(root: TreeNode) -> List[int]:
    visited = list()
    return postorder_DFS(root, visited)


def iterative_postorder_traversal(root: TreeNode) -> List[int]:
    visited = list()
    stack = list()
    current = root
    while current or stack:
        while current:
            visited.append(current.val)
            stack.append(current.left)
            current = current.right
        if stack:
            current = stack.pop()
    visited.reverse()
    return visited


# Binary Search Tree Iterator (Inorder)
class BSTIterator:
    # Complexity:
    #  T - O(1) Average (amortized) case (push + pop for every node = 2N * O(1) / N next() calls = O(1))
    #  S - O(h) - At all times stack only holds the leftmost nodes from a given root node
    def __init__(self, root: TreeNode):
        self.stack = list()
        self._inorder_left_traversal(root)

    def _inorder_left_traversal(self, node: TreeNode):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        """
        @return the next smallest number
        """
        current = self.stack.pop()
        if current.right:
            self._inorder_left_traversal(current.right)
        return current.val

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return bool(self.stack)

    def __iter__(self):
        return self

    def __next__(self):
        if self.hasNext():
            return self.next()
        else:
            raise StopIteration


def iterator_inorder_traversal(root: TreeNode) -> List[int]:
    bst_iterator = BSTIterator(root)
    bst_iter = iter(bst_iterator)
    return list(bst_iter)


def test(root: TreeNode, traversal: Callable, expected: List[int]):
    """Run test on given root, traversal method and validate output"""
    start = timer()
    output = traversal(root)
    end = timer()
    print(
        f"{traversal.__name__.title()}: {output == expected}, {output}, {end - start:.8f}"
    )


def main():
    node_7 = TreeNode(7)
    node_5 = TreeNode(5, None, node_7)
    node_2 = TreeNode(2)
    node_3 = TreeNode(3, node_2, node_5)
    node_0 = TreeNode(0)
    node_1 = TreeNode(1, node_0, node_3)
    test(node_1, recursive_inorder_traversal, [0, 1, 2, 3, 5, 7])
    test(node_1, iterative_inorder_traversal, [0, 1, 2, 3, 5, 7])
    test(node_1, iterator_inorder_traversal, [0, 1, 2, 3, 5, 7])
    test(node_1, recursive_preorder_traversal, [1, 0, 3, 2, 5, 7])
    test(node_1, iterative_preorder_traversal, [1, 0, 3, 2, 5, 7])
    test(node_1, recursive_postorder_traversal, [0, 2, 7, 5, 3, 1])
    test(node_1, iterative_postorder_traversal, [0, 2, 7, 5, 3, 1])
    test(node_1, iterative_levelorder_BFS_traversal, [[1], [0, 3], [2, 5], [7]])


if __name__ == "__main__":
    main()
