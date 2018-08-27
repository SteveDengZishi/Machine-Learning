#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 02:19:24 2018

@author: stevedeng
"""

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#define tensors and nodes
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

addition = tf.add(X, Y, name='addition')

#create session
with tf.Session() as session:
    result = session.run(addition, feed_dict={X:[1], Y:[4]})
    print(result)