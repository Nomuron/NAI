import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm


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

# Support Vector classifier
params = {'kernel': 'linear'}
support_vector_machine = svm.SVC(**params)

# Teaching the model
support_vector_machine.fit(X_train, y_train)

# Calculating predict values for test dataset
y_test_pred = support_vector_machine.predict(X_test)

# Evaluate classifier performance
class_names = ['Class-1', 'Class-2', 'Class-3']
print("\n" + "#"*40)
print("\nClassifier performance of wheat seeds support vector classifier model on training dataset\n")
print(classification_report(y_train, support_vector_machine.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance of wheat seeds support vector classifier model on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

# Showing plot
plt.show()
