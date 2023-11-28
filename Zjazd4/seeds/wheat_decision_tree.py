"""
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
numpy package
pandas package
pyplot from matplotlib
and some functions of sklearn-learn:
classification_report from sklearn.metrics
train_test_split from sklearn.model_selection
DecisionTreeClassifier from sklearn.tree

Pycharm or other IDE for Python
Link to install python: https://www.python.org/downloads/
To run script you need to run it from IDE .

==========================================
Decision tree
==========================================

Decision trees are a popular supervised learning technique used in machine learning
for both classification and regression tasks.
They organize data into a tree-like structure where each internal node represents a feature,
each branch denotes a decision based on that feature, and each leaf node holds a final outcome or prediction.
These trees make decisions by recursively splitting the dataset along the features,
selecting the most informative features at each step to maximize the homogeneity of the resulting subsets.

In provided example we are analyzing data about seeds. Those seeds are divided into 3 classes.
We are using 75% of data for training and 25% of data for testing.
We have used max depth of tree for 8
After learning algorythm and checking on testing data we are receiving information about algorythm precision.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Load input data
input_file = 'seeds_dataset.txt'
data = np.loadtxt(input_file, delimiter='\t')
X, y = data[:, :-1], data[:, -1]

# Separate input data into three classes based on labels
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])
class_3 = np.array(X[y == 3])

# Visualize input data
plt.figure()
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='black',
        edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
        edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_3[:, 0], class_3[:, 1], s=75, facecolors='white',
        edgecolors='black', linewidth=1, marker='v')
plt.ylabel('Area')
plt.xlabel('Perimeter')
plt.title('Input data')

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

# Decision Tree classifier
params = {'random_state': 0, 'max_depth': 8}
classifier = DecisionTreeClassifier(**params)

# Teaching the model
classifier.fit(X_train, y_train)

# Calculating predict values for test dataset
y_test_pred = classifier.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-1', 'Class-2', 'Class-3']
print("\n" + "#"*40)
print("\nClassifier performance of wheat seeds decision tree model on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance of wheat seeds decision tree model on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

# Showing plot
plt.show()
