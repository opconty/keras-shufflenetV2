#-*- coding:utf-8 -*-
#'''
# Created on 18-8-14 下午5:11
#
# @Author: Greg Gao(laygin)
#'''
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import keras

img_path = r'../images/a.png'
model_path = r'../weights/shufflenetv2_emotion_recogn.h5'
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]


image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (48, 48))
gray = gray.astype(np.float) / 255.
gray = img_to_array(gray)


model = load_model(model_path)


emo = model.predict(np.expand_dims(gray, 0))[0]

label = EMOTIONS[emo.argmax()]

cv2.putText(image, label, (3,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
cv2.imwrite('../images/a_emo.png', image)
cv2.imshow('emotion', image)

if cv2.waitKey(0):
    cv2.destroyAllWindows()