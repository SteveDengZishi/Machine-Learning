#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 01:09:34 2018

@author: stevedeng
"""

from keras.models import Sequential
from keras.layers import Dense, Activation

#using sequential model
model = Sequential([
            #Dense layer, alternatives are convolutional, pooling, recurrent...
            Dense(5, input_shape=(3,), activation='relu'),
            Dense(2, activation='softmax'),
        ])

model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled_training, Y_scaled_training, batch_size=10, epochs=20, shuffle=True, verbose=2)