# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:44:37 2017

@author: Arshad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:44:57 2017

@author: Arshad
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
                

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Featuring Scale
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Simple linear Regression in training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediting test data
y_pred = regressor.predict(X_test)

#Visulaizing data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salry vs Exprerince (Trainig seet)')
plt.xlabel ('Employee Year of Experience')
plt.ylabel ('Salary')
plt.show()


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salry vs Exprerince (Test seet)')
plt.xlabel ('Employee Year of Experience')
plt.ylabel ('Salary')
plt.show()
