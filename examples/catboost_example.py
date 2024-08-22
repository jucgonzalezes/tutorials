import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from catboost import Pool
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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

# def _plot_oblivious_tree(self, splits, leaf_values):
#         graph = {}

#         layer_size = 1
#         current_size = 0

#         for split_num in range(len(splits) - 1, -2, -1):
#             for node_num in range(layer_size):
#                 graph[current_size] = 
#                 if split_num >= 0:
#                     node_label = splits[split_num].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
#                     color = 'black'
#                     shape = 'ellipse'
#                     graph[current_size][1] = node_label
#                 else:
#                     node_label = leaf_values[node_num]
#                     color = 'red'
#                     shape = 'rect'
#                     graph[current_size][1] = node_label

#                 try:
#                     graph[current_size][1] = graph[current_size][1].decode("utf-8")
#                 except Exception:
#                     pass

#                 graph[current_size][0].append()
#                 graph.node(str(current_size), node_label, color=color, shape=shape)

#                 if current_size > 0:
#                     parent = (current_size - 1) // 2
#                     edge_label = 'Yes' if current_size % 2 == 0 else 'No'
#                     graph.edge(str(parent), str(current_size), edge_label)

#                 current_size += 1

#             layer_size *= 2

#         return graph

num_samples = 3_000

data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female', 'Other'], num_samples),
    'Occupation': np.random.choice(['Doctor', 'Engineer', 'Teacher', 'Artist'], num_samples),
    'Age': np.random.randint(20, 60, num_samples),
    'Income': np.random.randint(30000, 100000, num_samples),
    'Label': np.random.choice([0, 1], num_samples)
})

# Define categorical features
categorical_features = ['Gender', 'Occupation']

X = data.drop(columns='Label')
y = label=data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

# Convert to Pool
train_dataset = Pool(X_train, y_train, cat_features=categorical_features)
test_dataset = Pool(X_test, y_test, cat_features=categorical_features)

# Train CatBoost model
model = CatBoostClassifier(iterations=1, depth=4, learning_rate=0.1, loss_function='Logloss', verbose=False)
# model.fit(train_dataset)

grid = {'iterations': [1],
        'learning_rate': [0.03, 0.1, 1],
        'depth': [2, 4, 6, 8, 10],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}

model.grid_search(grid, train_dataset)

print(model.get_all_params())

print("\n SECTION BREAK --------------")
pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print("Testing performance")
print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))

tree_model = model.plot_tree(tree_idx=0, pool=train_dataset)

print(model.get_tree_leaf_counts)
print(model._get_tree_splits(0, train_dataset))
print(model._get_tree_leaf_values(0))
# print(model._get_tree_node_to_leaf(0))

tree_model.save("./examples/tree.dot")
model.save_model("./examples/model.json", format="json")
model.save_borders("./examples/borders")
# pool, _ = self._process_predict_input_data(pool, "plot_tree", thread_count=-1) if pool is not None else (None, None)

#         splits = self._get_tree_splits(tree_idx, pool)
#         leaf_values = self._get_tree_leaf_values(tree_idx)
#         if self._object._is_oblivious():
#             return self._plot_oblivious_tree(splits, leaf_values)
#         else:
#             step_nodes = self._get_tree_step_nodes(tree_idx)
#             node_to_leaf = self._get_tree_node_to_leaf(tree_idx)
#             return self._plot_nonsymmetric_tree(splits, leaf_values, step_nodes, node_to_leaf)