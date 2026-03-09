from DecisionTree import DecisionTree
import numpy as np

#Random Data
Y_train = np.array([[2], [4], [6], [8], [11], [13], [16], [19], [22], [25], [26], [27], [28], [29], [32], [34], [36], [38], [40]])
X_train = np.array([[0], [0], [0], [0], [5], [17], [100], [100], [100], [100], [65], [60], [55], [50], [45], [13], [0], [0], [0]])

model = DecisionTree(
    max_depth=5,
    min_samples_leaf=1,
    min_information_gain=0.0
)

model.train(X_train,Y_train)
model.print_tree()