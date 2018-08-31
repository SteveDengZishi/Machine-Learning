#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 01:09:34 2018

@author: stevedeng
"""

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
            Dense(32, input_shape=(10,), activation='relu'),
            Dense(2, activation='softmax'),
        ])