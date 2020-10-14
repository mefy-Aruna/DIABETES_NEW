# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:19:20 2020

@author: Aruna
"""


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
import h5py

numpy.random.seed(123)

dataset = numpy.loadtxt("mice.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]
x_train, x_validation, y_train, y_validation = train_test_split(    X, Y, test_size=0.20, random_state=100)

model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=10,validation_data=(x_validation, y_validation))


scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


model.save('model.h5')