#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:03:54 2018

@author: stevedeng
"""
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def readData():
    global X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing, X_scaler, Y_scaler, Y_training, Y_testing
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
    
    print("X_testing is : \n", X_testing[75:80])
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

def loadModel():
    with tf.Session() as session:
        
        #load previous saved model
        saver = tf.train.import_meta_graph('./models/trained_model.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('./models/'))
        
        graph = tf.get_default_graph()
        
        prediction = graph.get_tensor_by_name("output/prediction:0")
        X = graph.get_tensor_by_name('input/X:0')
        ## Now that the neural network is trained, let's use it to make predictions for our test data.
        # Pass in the X testing data and run the "prediciton" operation
        Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})
    
        # Unscale the data back to it's original units (dollars)
        Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)
    
        house_real_pricing = Y_testing[45:50]
        predicted_pricing = Y_predicted[45:50]
    
        print("The actual house price of House_1 was $\n{}".format(house_real_pricing))
        print("Our neural network predicted prices of $\n{}".format(predicted_pricing))
        
def main():
    readData()
    loadModel()

if __name__ == "__main__":
    main()    