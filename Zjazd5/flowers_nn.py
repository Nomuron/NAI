"""
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
numpy package
TensorFlow with some packages which should be imported in the code:
- Sequential model
- Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling


Pycharm or other IDE for Python
Link to install python: https://www.python.org/downloads/
To run script you need to run command "python flowers_nn.py"

==========================================
Neural Networks
==========================================

Neural networks are computing systems inspired by the human brain's structure and function.
They consist of interconnected nodes called neurons organized in layers.
Each neuron processes input data and transmits signals to neurons in the next layer.
Neural networks learn by adjusting connections between neurons based on example data.

In provided example we are analyzing photos of flowers.
This code demonstrates how to prepare an image dataset, build a CNN model for classification,
train the model, evaluate its performance, and make predictions on specific images within the dataset.

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

import pathlib

# creating temporary directory on local machine, download dataset and extract it there
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

# generate train sub dataset on a base of pictures in directory
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

# generate test/validation sub dataset on a base of pictures in directory
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

# get label names
class_names = train_ds.class_names

# define model
model = Sequential()
model.add(Rescaling(1./255, input_shape=(180, 180, 3)))
model.add(Conv2D(16, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)),
model.add(Dense(len(class_names), activation='softmax'))

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fitting model
model.fit(train_ds, epochs=10, validation_data=val_ds)

# evaluate model
score = model.evaluate(val_ds, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predicting_record = 63
for images, labels in val_ds.take(predicting_record + 1):
    # converting batches to NumPy arrays
    images_np = images.numpy()
    labels_np = labels.numpy()

    # accessing specific image and label
    specific_image_np = images_np[predicting_record % 32]
    specific_label_np = labels_np[predicting_record % 32]

    # printing information about expected label
    print('Predicting rekord number : ', predicting_record)
    print('Expected label : ', specific_label_np)

    # predict result for specific record
    predict = model.predict(np.expand_dims(specific_image_np, axis=0))
    print('Predicted label: ', np.argmax(predict))
    break
