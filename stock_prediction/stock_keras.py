#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:28:06 2018

@author: stevedeng
"""
import pandas as pd

def data_to_html():
    #Read the dataset using pandas
    data_table = pd.read_csv("data_stocks.csv")
    html = data_table[:100].to_html()
    
    #with open do not need a close statement
    with open("data.html", "w") as outfile:
        outfile.write(html)

data_to_html()