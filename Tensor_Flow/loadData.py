#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:02:37 2018

@author: stevedeng
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
number_of_inputs = 63
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

def readData():
    global X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing, X_scaler, Y_scaler
    # Read csv data into dataframes using pandas
    # Load training data set from CSV file
    df = pd.read_csv("house_data.csv")

    del df["house_number"]
    del df["unit_number"]
    del df["street_name"]
    del df["zip_code"]
    
    #Replace categorical data with one-hot encoded data(order is not meaningful)
    features_df = pd.get_dummies(df, columns=['garage_type','city'])
    
    #Remove the final sale price
    del features_df['sale_price']
    
    #Create the X and Y arrays and convert them to numpy matrix
    X = features_df.as_matrix()
    Y = df[['sale_price']].as_matrix()
    
    #Split the dataset in 70% training and 30% testing
    X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    #print("X_training is : \n", X_training)
    #print("Y_training is: \n", Y_training)
    
    # All data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well. Difference in scale among different colums
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    
    # Scale both the training inputs and outputs
    #print(Y_training)
    X_scaled_training = X_scaler.fit_transform(X_training)
    Y_scaled_training = Y_scaler.fit_transform(Y_training)
    
    #print("The scale on X_data is: \n", X_scaler.scale_, "\nWith adjustments of: \n", X_scaler.min_)
    #print("\nThe scale on Y_data is: \n", Y_scaler.scale_, "\nWith adjustments of: \n", Y_scaler.min_)
    #print("\nNote: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))
    
    # It's very important that the training and test data are scaled with the same scaler.
    X_scaled_testing = X_scaler.transform(X_testing)
    Y_scaled_testing = Y_scaler.transform(Y_testing)
    
    
    #print("\nThe size of the training/testing datasets are")
    #print(X_scaled_training.shape, Y_scaled_training.shape)
    #print(X_scaled_testing.shape, Y_scaled_testing.shape)
    
def trainModel():
    global number_of_inputs, number_of_outputs, learning_rate, training_epochs, display_step, layer_1_nodes, layer_2_nodes, layer_3_nodes
    
    # Section_1 define neural network layers
    # Input Layer
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
        
    # Layer 1
    with tf.variable_scope('layer_1'):
        # Use tf.get_variable to insert a variable and specify the shape of the matrix
        weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)
    
    # Layer 2
    with tf.variable_scope('layer_2'):
        # Use tf.get_variable to insert a variable and specify the shape of the matrix
        weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)
    
    # Layer 3
    with tf.variable_scope('layer_3'):
        # Use tf.get_variable to insert a variable and specify the shape of the matrix
        weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)
    
    # Output Layer
    with tf.variable_scope('output'):
        # Use tf.get_variable to insert a variable and specify the shape of the matrix
        weights = tf.get_variable(name='weights4', shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
        # Using relu and matrix multiplication to define the activation function
        prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)
        
    
    # Section_2 define cost function in order to measure the prediction accuracy neural network
    with tf.variable_scope('cost'):
        Y = tf.placeholder(tf.float32, shape=(None, 1))
        # Need to calculate the mean square error of the value predicted
        cost = tf.reduce_mean(tf.squared_difference(prediction, Y))
    
    # Section_3 define Optimizer function to run optimize on the neural network
    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
    #logging to show on tensor board
    with tf.variable_scope('logging'):
        tf.summary.scalar('current_cost', cost)
        tf.histogram('predicted_value', prediction)
        log = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    # Training Part using Session
    # Create a tensorflow session to run operation
    with tf.Session() as session:
        # Initialize all variables and layers using global initializer
        session.run(tf.global_variables_initializer())
        
        #writing logs
        training_writer = tf.summary.FileWriter('./logs/training', session.graph)
        testing_writer = tf.summary.FileWriter('./logs/testing', session.graph)
        
        # Iteratively train model to fit the model
        for i in range(training_epochs):
            #run optimizer
            session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            
            #run cost functions to get training & testing costs
            training_cost, training_log = session.run([cost, log], feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_log = session.run([cost, log], feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
            
            #log functions to write log to files
            #tensorboard --logdir=logs
            training_writer.add_summary(training_log, i)
            training_writer.flush()
            testing_writer.add_summary(testing_log, i)
            testing_writer.flush()
            
            print("Training pass: {}".format(i))
            print("The training cost is {} and the testing cost is {}".format(training_cost, testing_cost))
            
        final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
        print("Training Completed!")
        print("The final training cost is {} and the fianl testing cost is {}".format(final_training_cost, final_testing_cost))
        save_path = saver.save(session, './models/trained_model.ckpt')
        print("Model saved at: ", save_path)
        
def main():
    readData()
    trainModel()

if __name__ == "__main__":
    main()    