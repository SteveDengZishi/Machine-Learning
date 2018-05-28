#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:14:49 2018

@author: stevedeng
"""

import pandas as pd

def data_to_html():
    # Read the dataset using pandas
    data_table = pd.read_csv("house_data.csv")
    html = data_table[:100].to_html()
    
    with open("data.html", "w") as outfile:
        outfile.write(html)
        
def train_model():
    df = pd.read_cvs("house_data.csv")
    
    # Delete irrelevant feature columns
    del df["house_number"]
    del df["unit_number"]
    del df["street_name"]
    del df["zip_code"]

def main():
    
    

if __name__ == "__main__":
    main()    