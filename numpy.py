#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:41:02 2018

@author: stevedeng
"""
import numpy as np

#use numpy array to run SIMD instruction
sizes = np.array([100.0, 298.3, 232.53, 458.90])

sizes = sizes * 0.3
print(sizes)

