import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import math
import tensorflow as tf
import cv2
import cvlib as cv

model = load_model("Model5.h5")
model.summary()

emotion_dict = {0: "Angry", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprised"}

ic = cv2.imread('test_photos\\happy_group.jpgq')
ic = cv2.resize(ic, (1280, 720))
gray = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)

face, confidence = cv.detect_face(ic)
for idx, f in enumerate(face):
    # get corner points of face rectangle
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]

    # draw rectangle over face
    cv2.rectangle(ic, (startX,startY), (endX,endY), (0,255,0), 2)

    # crop the detected face region
    roi_gray = gray[startY:endY, startX:endX]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    label = "{}: {:.2f}%".format(emotion_dict[maxindex], prediction[0][maxindex]*100)
    cv2.putText(ic ,label, (startX + 20, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2,cv2.LINE_AA)

cv2.imshow('image', ic)
cv2.waitKey(0)
