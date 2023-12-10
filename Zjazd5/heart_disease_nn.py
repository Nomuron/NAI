"""
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
numpy package
pandas package
TensorFlow with some packages which should be imported in the code:
- Sequential model
- Conv2D, MaxPooling2D, Flatten, Dense, Dropout
train_test_split from sklearn.model_selection


Pycharm or other IDE for Python
Link to install python: https://www.python.org/downloads/
To run script you need to run command "python heart_disease_nn.py"

==========================================
Neural Networks
==========================================

Neural networks are computing systems inspired by the human brain's structure and function.
They consist of interconnected nodes called neurons organized in layers.
Each neuron processes input data and transmits signals to neurons in the next layer.
Neural networks learn by adjusting connections between neurons based on example data.

This code builds a simple neural network model to predict the presence of heart disease based on various input features
and evaluates its performance on a held-out test set,
also demonstrating how to make predictions for a specific record in the dataset.


"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# load input data
data = pd.read_csv('data/heart.csv')

# split columns into inputs and labels
X, y = data.iloc[:, 0:-1], data.iloc[:, -1]

# split datasets to test and train datasets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=5)

# define model
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1)

# evaluate second model
score1 = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score1[0])
print("Test accuracy:", score1[1])

# define specific record for test of prediction
predicting_record = 2
print('Predicting rekord number : ', predicting_record)
print('Expected label : ', y_test.iloc[predicting_record])

# predict result for specific record
predict = model.predict(np.expand_dims(X_test.iloc[predicting_record], axis=0))
print('Predicted label: ', np.argmax(predict))
