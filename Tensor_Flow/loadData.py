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
    #double [[]] is to make every row a standalone list
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
    
    # All data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well. Difference in scale among different colums
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    
    # Scale both the training inputs and outputs
    X_scaled_training = X_scaler.fit_transform(X_training)
    Y_scaled_training = Y_scaler.fit_transform(Y_training)
    
    print("The scale on X_data is: \n", X_scaler.scale_, "\nWith adjustments of: \n", X_scaler.min_)
    print("\nThe scale on Y_data is: \n", Y_scaler.scale_, "\nWith adjustments of: \n", Y_scaler.min_)
    print("\nNote: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))
    
    # It's very important that the training and test data are scaled with the same scaler.
    X_scaled_testing = X_scaler.transform(X_testing)
    Y_scaled_testing = Y_scaler.transform(Y_testing)
    
    
    print("\nThe size of the training/testing datasets are")
    print(X_scaled_training.shape, Y_scaled_training.shape)
    print(X_scaled_testing.shape, Y_scaled_testing.shape)
    
def main():
    readData()


if __name__ == "__main__":
    main()    