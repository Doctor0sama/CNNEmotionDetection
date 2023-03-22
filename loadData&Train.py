import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, to_categorical
from tensorflow.python.keras.callbacks import ReduceLROnPlateau


dataset = []
labels = []

imgpath = 'Dataset'
for path, _, files in os.walk(imgpath):
    for file in files:
        image = cv2.imread(path + '\\' + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        dataset.append(image)

        label = path.split(os.path.sep)[-1]
        labels.append(label)

print(len(dataset))
print(len(labels))


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(encoded_Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)
print(dummy_y.shape)
print(labels)

X = np.asarray(dataset)
Y = np.asarray(dummy_y)

X_train, X_notTrain, y_train, y_notTrain = train_test_split(X, Y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_notTrain, y_notTrain, test_size=0.5, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

print(X_train.shape[0])

X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1,48, 48, 1)
X_val = X_val.reshape(-1,48, 48, 1)


#Creating model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto',
                                   cooldown=0, min_lr=1e-4, verbose=2)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=64,epochs=50,callbacks=[reduce_lr], verbose=2, validation_data=(X_val, y_val))
model.save("Model5.h5")

print("\nEvaluating : ")
loss, accuracy = model.evaluate(X_test, y_test, batch_size=16)
print()
print("Final Accuracy : ", accuracy)
print("Final Loss : ", loss)

predicted_y = model.predict(X_test, batch_size=16)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()