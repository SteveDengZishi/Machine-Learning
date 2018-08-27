#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 02:19:24 2018

@author: stevedeng
"""

import os
import tensorflow as tf

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#define tensors
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')