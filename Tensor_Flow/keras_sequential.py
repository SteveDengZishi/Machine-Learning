#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 01:09:34 2018

@author: stevedeng
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard

#using sequential model
def model():
    global model
    model = Sequential([
                #Dense layer, alternatives are convolutional, pooling, recurrent...
                Dense(50, input_shape=(63,), activation='relu'),
                Dense(100, activation='relu'),
                Dense(50, activation='relu'),
                Dense(1)
            ])

def train():
    global X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing
    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model.compile(optimizer="adam", loss='mean_squared_error')
    
    #tensorboard --logdir=Logs  --port=8088  --host=localhost
    model.fit(X_scaled_training, Y_scaled_training, epochs=100, shuffle=True, verbose=2, callbacks=[tbCallBack])
    
    test_error_rate = model.evaluate(X_scaled_testing, Y_scaled_testing, verbose=2)
    print("The mean squared difference for testing data is {}".format(test_error_rate))
    
    expected_val = Y_testing[45:50]
    predicted_val = Y_scaler.inverse_transform(model.predict(X_scaled_testing)[45:50])
    
    model.save("./models/trained_model.h5")
    
    print("The expected housing prices are: \n", expected_val)
    print("The predicted prices from the model are: \n", predicted_val)
    
def readData():
    global X_scaled_training, Y_scaled_training, X_scaled_testing, Y_scaled_testing, X_scaler, Y_scaler, Y_testing
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
    
    
def main():
    readData()
    model()
    train()
    
if __name__ == "__main__":
    main()    