#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:41:43 2020

@author: ebin
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import sklearn as skl
import os
import sys

#reading the dataset and creating a dataframe
melbone_orig_dataframe = pd.read_csv('/Users/synup/Desktop/Ebin/machine_learning/datasets/melb_data.csv')

#making the deep copy of oroginal dataset for test case1
approach1_data = melbone_orig_dataframe.copy(deep=True)

#checking the attributes of data
approach1_data.index
approach1_data.columns
approach1_data.size
approach1_data.shape
approach1_data.memory_usage()
approach1_data.head()
approach1_data.tail()
approach1_data.get_dtype_counts()
approach1_data.dtypes()
approach1_data.info()

#splitting the training and test data
import pandas as pd
from sklearn.model_selection import train_test_split
data1 = approach1_data
y = data1.Price #target
#attributes
melb_predictors = data1.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])
#dividing dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

#function to measure the quality of each approach
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#the function
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#approach1 drop columns with missing values

#drop columns with missing values
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


#approach2 simple imputation
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    