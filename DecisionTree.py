import numpy as np
from collections import Counter
from TreeNode import TreeNode


class DecisionTree:
    def __init__(
        self,
        max_depth=5,
        min_samples_leaf=1,
        min_information_gain=0.0,
        numb_of_features_splitting=None,
        amount_of_say=None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say
        
    def _class_probabilities(self, labels: list) -> list:
        total_count = len(labels) 
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _entropy(self, y):
        """
        Calculates the entropy for a given set of labels.
        """
        _, counts = np.unique(y, return_counts=True)  # "_" is an empty variable
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])

        return entropy
        
    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:        
        mask_below_threshold = data[:, feature_idx] < feature_val #Boolean Mask
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold] # "~" symbol inverts the boolean mask!!
        return group1, group2
        
    def _data_entropy(self, labels: list) -> float:
        return self._entropy(labels)
        
    def _partition_entropy(self, subsets: list)  -> float:
        """subsets = list of label lists"""
        total_count = sum(len(subset) for subset in subsets)
        if total_count == 0:
            return 0
        return sum([self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets])

    def _best_split(self, data: np.array) -> tuple:
        min_split_entropy = 1e9
        min_entropy_feature_idx = None
        min_entropy_feature_val = None
        group1_min, group2_min = None, None

        for idx in range(data.shape[1] - 1):
            feature_val = np.median(data[:, idx])
            g1, g2 = self._split(data, idx, feature_val)
            split_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
            if split_entropy < min_split_entropy:
                min_split_entropy = split_entropy
                min_entropy_feature_val = feature_val
                min_entropy_feature_idx = idx
                group1_min, group2_min = g1, g2

        return (
            group1_min,
            group2_min,
            min_entropy_feature_idx,
            min_entropy_feature_val,
            min_split_entropy,
        )
        
    def _find_label_probs(self, data: np.array) -> np.array:
        labels_as_integers = data[:, -1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == label)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index)/total_labels
        return label_probabilities
        
    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        """
        Recursive, depth first tree creation algorithm
        """
        if current_depth > self.max_depth:
            return None
            
        #find Best Split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._best_split(data)
        
        #find label probs for the node
        label_probabilites = self._find_label_probs(data)
        
        #Calculate Information Gain
        node_entropy = self._entropy(data[:, -1])
        information_gain = node_entropy - split_entropy
        
        #create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilites, information_gain)
        
        if split_1_data is None or split_2_data is None:
            return node

        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        
        elif information_gain < self.min_information_gain:
            return node
            
        current_depth += 1
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)
        
        return node
        
    def _predict_one_sample(self, X: np.array) -> np.array:
        node = self.tree 
        pred_probs = node.prediction_probs
        
        while node:
            pred_probs = node.prediction_probs
            if node.left is None or node.right is None:
                break
            if X[node.feature_idx] < node.feature_val:
                node = node.left
                
            else:
                node = node.right
                
        return pred_probs
        
    def train(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1,1))), axis=1)
        
        self.tree = self._create_tree(data=train_data, current_depth=0)
        
        self.feature_importances = {i: 0.0 for i in range(X_train.shape[1])}
        self._calculate_feature_importance(self.tree)
        total_importance = sum(self.feature_importances.values())
        if total_importance > 0:
            self.feature_importances = {k: v / total_importance for k, v in self.feature_importances.items()}
    
    def predict_proba(self, X_set: np.array) -> np.array:
        return np.array([self._predict_one_sample(X) for X in X_set])
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)
            
    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)
        
    def _calculate_feature_importance(self, node):
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)