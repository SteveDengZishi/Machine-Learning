#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:02:37 2018

@author: stevedeng
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X_scaled_training = None
Y_scaled_training = None
X__scaled_testing = None
Y_scaled_testing = None

X_scaler = None
Y_scaler = None

# Define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

def readData():
    global X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing, X_scaler, Y_scaler
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
    
def model():
    global number_of_inputs, number_of_outputs, learning_rate, training_epochs, display_step, layer_1_nodes, layer_2_nodes, layer_3_nodes
    
    # Input Layer
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float, shape=(None, number_of_inputs))
        
    # Layer 1
    with tf.variable_scope('layer_1'):
        # Use tf.get_variable to insert a variable and specify the shape of the matrix
        weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)
    
    # Layer 2
    
    
def main():
    readData()


if __name__ == "__main__":
    main()    