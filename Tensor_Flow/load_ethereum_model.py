#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:16:29 2018

@author: stevedeng
"""

import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def readData():
    global X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing, X_scaler, Y_scaler, Y_testing
    # Read csv data into dataframes using pandas
    # Load training data set from CSV file
    df = pd.read_csv("ethereum.csv", sep=";")
    del df['date']
    df.dropna(inplace=True)
    
    #Replace categorical data with one-hot encoded data(order is not meaningful)
    features_df = df.copy()
    #Remove the final sale price
    del features_df['price(USD)']
    del features_df['marketcap(USD)']
    
    #Create the X and Y arrays and convert them to numpy matrix
    X = features_df.as_matrix()
    Y = df[['price(USD)']].as_matrix()
    
    #Split the dataset in 70% training and 30% testing
    X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    # All data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well. Difference in scale among different colums
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale both the training inputs and outputs
    #print(Y_training)
    X_scaled_training = X_scaler.fit_transform(X_training)
    Y_scaled_training = Y_scaler.fit_transform(Y_training)
    
    # It's very important that the training and test data are scaled with the same scaler.
    X_scaled_testing = X_scaler.transform(X_testing)
    Y_scaled_testing = Y_scaler.transform(Y_testing)

def loadModel():
    global model
    model = load_model("./models/trained_ethereum_model.h5")
    
def makePrediction():
    
    expected_val = Y_testing[70:78]
    predicted_val = Y_scaler.inverse_transform(model.predict(X_scaled_testing)[70:78])
    print("The expected ethereum prices are: \n", expected_val)
    print("The predicted prices from the model are: \n", predicted_val)
    
def main():
    readData()
    loadModel()
    makePrediction()
    
if __name__ == "__main__":
    main()  