#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:02:37 2018

@author: stevedeng
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def readData():
    # Read csv data into dataframes using pandas
    # Load training data set from CSV file
    training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

    # Pull out columns for X (data to train with) and Y (value to predict)
    X_training = training_data_df.drop('total_earnings', axis=1).as_matrix()
    Y_training = training_data_df[['total_earnings']].as_matrix()
    
    # Using the following way will pass by reference and lose column permanently
    # X_df = training_data_df
    # del X_df['total_earnings']
    # X_matrix = X_df.as_matrix()
    # print(training_data_df.as_matrix())
    
    # Load testing data set from CSV file
    test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)
    
    # Pull out columns for X (data to train with) and Y (value to predict)
    X_testing = test_data_df.drop('total_earnings', axis=1).values
    Y_testing = test_data_df[['total_earnings']].values
    
    #print("X_training is : \n", X_training)
    #print("Y_training is: \n", Y_training)
    
    
def main():
    readData()


if __name__ == "__main__":
    main()    