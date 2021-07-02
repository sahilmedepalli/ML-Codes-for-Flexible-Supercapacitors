# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:11:30 2021

@author: Nicholas
"""

from sklearn.ensemble import RandomForestRegressor
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x = []
y = []
z = []

file_name = str(input('Name of CSV file:'))
with open(file_name + '.csv', 'r') as datfile:
    X = [[float(x), float(y), float(z)] for x, y, z in csv.reader(datfile, delimiter= ',')]
    for row_var in X:
        if row_var[0] != 'Cycle no.' and row_var[1] != 'Ratio no.' and row_var[2] != 'Capacitance':
            x.append(float(row_var[0]))
            y.append(float(row_var[1]))
            z.append(float(row_var[2]))
           
#Creates Training and Testing Data Split
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=50, random_state=0)            
            
#Defines the Model
model = RandomForestRegressor(n_estimators = 1000, random_state = 0)

#Fits the Model on the Whole Dataset
model.fit(X_train,z_train)

z_pred = model.predict(X_test)

#Plots the Prediction Points (red dots) vs Actual Data (blue line)
plt.plot(z, z, 'b-')
plt.plot(z_test, z_pred,'ro')
plt.legend(["Real Data", "Predicted"])

#Calculates MSE for Comparing Algorithms

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(z_test, z_pred)
print('MSE:', MSE)