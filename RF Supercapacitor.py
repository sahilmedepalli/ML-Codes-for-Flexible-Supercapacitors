# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:43:51 2021

@author: Nicholas
"""

from sklearn.ensemble import RandomForestRegressor
import csv
import matplotlib.pyplot as plt

x = []
y = []

file_name = str(input('Name of CSV file:'))
with open(file_name + '.csv', 'r') as datfile:
    X = [[float(x), float(y)] for x, y in csv.reader(datfile, delimiter= ',')]
    for row_var in X:
        if row_var[0] != 'Capacitance' and row_var[1] != 'Cycle no.':
            x.append(float(row_var[0]))
            y.append(float(row_var[1]))
            
print(X)
print(y)

#Defines the Model
model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
#Fits the Model on the Whole Dataset
model.fit(X,y)
yhat = model.predict(X)
print(yhat[0:199])

#Plots the Prediction Points (red dots) vs Actual Data (blue line)
plt.plot(x, y, 'b-')
plt.plot(x, yhat,'ro')
plt.show()

cycleno = int(input('Cycle no.?'))
print('Prediction:', yhat[cycleno-1])



