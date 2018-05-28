#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:14:49 2018

@author: stevedeng
"""

import pandas

# Read the dataset using pandas
data_table = pandas.read_csv("house_data.csv")
html = data_table[:100].to_html()

with open("data.html", "w") as outfile:
    outfile.write(html)