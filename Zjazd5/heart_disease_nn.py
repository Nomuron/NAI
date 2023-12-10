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
