#Part1
import numpy as np
import matplotlib.pyplot as plt
import math

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

#Display of first 5 elements

print("First the 5 first elements of X_train:\n", X_train[:5]) 
print("Type of X_train:",type(X_train)) 
 
print("First the 5 first elements of y_train:", y_train[:5]) 
print("Type of y_train:",type(y_train))

# Dimensions verification
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape) 
print ('Number of training examples (m):', len(X_train))

#Computing Entropy
def compute_entropy(y):
    entropy = 0.
    if len(y) != 0:
        p1=len(y[y==1])/len(y)
        if p1 !=1 and p1 !=0:
            entropy = -p1*math.log2(p1) -(1-p1)*math.log2(1-p1)
    return entropy

print("Entropy at root node: ", compute_entropy(y_train))

#Splitting the dataset
def split_dataset(X, node_indices, feature): 
    left_indices = [] 
    right_indices = []
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

#Entropy Verification
# Case 1 
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature = 0
 
left_indices, right_indices = split_dataset(X_train, root_indices, feature) 
 
print("CASE 1:") 
print("Left indices: ", left_indices) 
print("Right indices: ", right_indices) 
 
# Case 2 
root_indices_subset = [0, 2, 4, 6, 8] 
left_indices, right_indices = split_dataset(X_train, root_indices_subset, 
feature) 
 
print("CASE 2:") 
print("Left indices: ", left_indices) 
print("Right indices: ", right_indices)


#Computing information gain
def compute_information_gain(X, y, node_indices, feature): 

    left_indices, right_indices = split_dataset(X, node_indices, feature) 
 
    X_node, y_node = X[node_indices], y[node_indices] 
    X_left, y_left = X[left_indices], y[left_indices] 
    X_right, y_right = X[right_indices], y[right_indices] 
    information_gain = 0 
    information_gain = compute_entropy(y_node)-(len(X_left)/len(X_node))*compute_entropy(y_left) - (len(X_right)/len(X_node))*compute_entropy(y_right)

    return information_gain

#Info gain verification
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)
info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)
info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)


#Best tree split
def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = 0
    for feature in range(num_features):
        if compute_information_gain(X, y, node_indices, feature) > compute_information_gain(X, y, node_indices, best_feature):
            best_feature = feature
    return best_feature

#Split verfication
best_feature = get_best_split(X_train, y_train, root_indices) 
print("Best feature to split on: %d" % best_feature)

#Tree construction
tree = [] 
 
def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    # Maximum depth reached - stop splitting
    if current_depth == max_depth: 
        formatting = " " * current_depth + "-" * current_depth 
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return 
 
    # Otherwise, get best split and split the data 
    # Get the best feature and threshold at this node 
    best_feature = get_best_split(X, y, node_indices) 
 
    formatting = "-" * current_depth 
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
 
    # Split the dataset at the best feature 
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1)


#Tree verification
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)




























