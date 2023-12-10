"""
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
seaborn package
numpy package
pyplot from matplotlib
TensorFlow with some packages which should be imported in the code:
- fashion_mnist dataset
- Sequential model
- Conv2D, MaxPooling2D, Flatten, Dense, Dropout layers
- confusion_matrix

Pycharm or other IDE for Python
Link to install python: https://www.python.org/downloads/
To run script you need to run command "python clothes_nn.py"

==========================================
Neural Networks
==========================================

Neural networks are computing systems inspired by the human brain's structure and function.
They consist of interconnected nodes called neurons organized in layers.
Each neuron processes input data and transmits signals to neurons in the next layer.
Neural networks learn by adjusting connections between neurons based on example data.

In provided example we are analyzing photos of clothes.
This code demonstrates the entire pipeline of training a neural network, evaluating its performance,
visualizing the confusion matrix, and making predictions for specific records in the Fashion-MNIST dataset.

"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.ops.confusion_matrix import confusion_matrix

# load fashion_mnist data from tensorflow
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# normalize data to range between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# define model
model = Sequential()
model.add(Flatten(input_shape=(28, 28))),
model.add(Dense(128, activation='relu')),
model.add(Dropout(0.5)),
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fitting model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# creating prediction for whole test sub dataset
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# create confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

# plot and show confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# define specific record for test of prediction
predicting_record = 54
print('Predicting rekord number : ', predicting_record)
print('Expected label : ', y_test[predicting_record])

# predict result for specific record
predict = model.predict(np.expand_dims(X_test[predicting_record], axis=0))
print('Predicted label: ', np.argmax(predict))
