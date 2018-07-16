# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:36:32 2018

@author: kishore
"""

# Logistic Regression

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

#importing the dataset 
dataset_train = pd.read_csv('criminal_train.csv')
dataset_test = pd.read_csv('criminal_test.csv')

X_train = dataset_train.iloc[:,1:71].values
Y_train = dataset_train.iloc[:,71].values


X_test = dataset_test.iloc[:,1:71].values


# Scaling the dataset 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

df = pd.DataFrame(y_pred)
df.to_csv('y_predReg.csv')



