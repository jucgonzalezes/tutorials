def find_leaf_paths(tree, node, current_path, all_paths):
    current_path.append(node)

    if node not in tree or not tree[node]:  # This is a leaf node
        all_paths.append(current_path.copy())
    else:
        for child in tree[node]:
            find_leaf_paths(tree, child, current_path, all_paths)

    current_path.pop()  # Backtrack to explore other paths

def get_all_leaf_paths(tree, root):
    all_paths = []
    find_leaf_paths(tree, root, [], all_paths)
    return all_paths

# Example usage:
if __name__ == "__main__":
    # Constructing the tree as a dictionary
    tree = {
        "root": ["child1", "child2"],
        "child1": ["grandchild1", "grandchild2"],
        "child2": ["child3"],
        "child3": ["grandchild3", "grandchild4"],
        "grandchild1": [],
        "grandchild2": [],
        "grandchild3": [],
        "grandchild4": []
    }

    # Getting all leaf paths
    paths = get_all_leaf_paths(tree, "root")
    for path in paths:
        print(" -> ".join(path))