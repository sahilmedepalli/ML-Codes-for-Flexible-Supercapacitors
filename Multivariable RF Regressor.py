# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:11:30 2021

@author: Nicholas
"""

from sklearn.ensemble import RandomForestRegressor
import csv
import matplotlib.pyplot as plt

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
            
#Defines the Model
model = RandomForestRegressor(n_estimators = 1000, random_state = 0)
#Fits the Model on the Whole Dataset
model.fit(X,z)
zhat = model.predict(X)
print(zhat[0:199])

#Plots the Prediction Points (red dots) vs Actual Data (blue line)
plt.plot(x, z, 'b-')
plt.plot(x, zhat,'ro')
plt.legend(["Real Data", "Predicted"])
plt.show()

cycleno = int(input('Cycle no.?'))
print('Prediction w/ RF:', zhat[cycleno-1])

#Calculates R Squared for Comparing Algorithms
import numpy

corr_matrix = numpy.corrcoef(z, zhat)
corr = corr_matrix[0,1]
R_sq = corr**2
 
print('R squared value for RF:' , R_sq)