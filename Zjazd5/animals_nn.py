"""
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
numpy package
TensorFlow with some packages which should be imported in the code:
-cifar10 dataset
-Sequential model
-Conv2D, MaxPooling2D, Flatten, Dense, Dropout layers
-to_categorical utils

Pycharm or other IDE for Python
Link to install python: https://www.python.org/downloads/
To run script you need to run command "python animals_nn.py"

==========================================
Neural Networks
==========================================

Neural networks are computing systems inspired by the human brain's structure and function.
They consist of interconnected nodes called neurons organized in layers.
Each neuron processes input data and transmits signals to neurons in the next layer.
Neural networks learn by adjusting connections between neurons based on example data.

In provided example we are analyzing photos of animals and some vehicles.
In this implementation we created 2 architectures of neural network .
First one is less complex. Second one is more complex.
After some tests we can say that second model gives better results. Both overall results(%) and specific predictions.
"""
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# load cifar10 data from tensorflow
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize data to range between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# change labels value from scalar to vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# define first, less complex model
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dense(10, activation='softmax'))

# define second, more complex model
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10, activation='softmax'))

# compiling both models
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# define batch size and numbers of epochs
batch_size = 64
epochs = 10

# fitting both models
history1 = model1.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
history2 = model2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# evaluate first model
score = model1.evaluate(X_test, y_test, verbose=0)
print("Test model 1 loss:", score[0])
print("Test model 1 accuracy:", score[1])

# evaluate second model
score1 = model2.evaluate(X_test, y_test, verbose=0)
print("Test model 2 loss:", score1[0])
print("Test model 2 accuracy:", score1[1])

# define specific record for test of prediction
predicting_record = 44
print('Predicting rekord number : ', predicting_record)
print('Expected label : ', np.argmax(y_test[predicting_record]))

# predict results for specific record from two models
predict1 = model1.predict(np.expand_dims(X_test[predicting_record], axis=0))
predict2 = model2.predict(np.expand_dims(X_test[predicting_record], axis=0))
print('Predicted by model 1 label: ', np.argmax(predict1))
print('Predicted by model 2 label: ', np.argmax(predict2))
