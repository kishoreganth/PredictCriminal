# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:06:35 2018

@author: kisho
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the dataset 
training_set = pd.read_csv('criminal_train.csv')
test_set = pd.read_csv('criminal_test.csv')

X_train = training_set.iloc[: ,1:71]
Y_train = training_set.iloc[: , 71]

X_test = test_set.iloc[:, 1:]




# Fitting the multiple Linear Regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predicting the test set 
y_pred = regressor.predict(X_test)
y_pred = (y_pred> 0.5)



# building the optimal model using backward elimination 
import statsmodels.formula.api as smf 
import statsmodels.api as sm
X = np.append(arr = np.ones((45718,70)).astype(int), values = X_train, axis = 1)

# Summary of the code 
regressor_OLS = sm.OLS(endog = Y_train, exog = X).fit()
regressor_OLS.summary()

