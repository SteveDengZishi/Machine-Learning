#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:14:49 2018

@author: stevedeng
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

def data_to_html():
    #Read the dataset using pandas
    data_table = pd.read_csv("house_data.csv")
    html = data_table[:100].to_html()
    
    #with open do not need a close statement
    with open("data.html", "w") as outfile:
        outfile.write(html)
    
def train_model():
    df = pd.read_cvs("house_data.csv")
    
    #Delete irrelevant feature columns
    #Dataframe works as a python dictionary
    del df["house_number"]
    del df["unit_number"]
    del df["street_name"]
    del df["zip_code"]
    
    #Replace categorical data with one-hot encoded data(order is not meaningful)
    features_df = pd.get_dummies(df, columns=['garage_type','city'])
    
    #Remove the final sale price
    del features_df['sales_price']
    
    #Create the X and Y arrays and convert them to numpy matrix
    X = features_df.as_matrix()
    Y = df['sales_price'].as_matrix()
    
    #Split the dataset in 70% training and 30% testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    #Calling sklearn Regressor using gradient boosting algorithm
    #The parameters are used to specify the decision tree parameters
    model = ensemble.GradientBoostingRegressor(n_estimator=1000, learning_rate=0.1, max_depth=6, min_samples_leaf=9,\
                                               max_features=0.1, loss='huber')
    #Use your data to train the model
    model.fit(X_train, Y_train)
    
    #Save model
    joblib.dump(model, 'trained_house_classifier_model.pkl')
    
    
def main():
    data_to_html()
    train_model()
    
    
if __name__ == "__main__":
    main()    