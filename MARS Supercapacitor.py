# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:29:03 2021

@author: Ronnie
"""

#MARS Implemented with Lasso

import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

#Collects the Data from the CSV File
x_ = []
y_ = []
z_ = []

file_name = str(input('Name of CSV file:'))
with open(file_name + '.csv', 'r') as datfile:
    X = [[float(x), float(y), float(z)] for x, y, z in csv.reader(datfile, delimiter= ',')]
    for row_var in X:
        if row_var[0] != 'Cycle no.' and row_var[1] != 'Ratio no.' and row_var[2] != 'Capacitance':
            x_.append(float(row_var[0]))
            y_.append(float(row_var[1]))
            z_.append(float(row_var[2]))

x = pd.Series(x_)   
z = pd.Series(z_)

size = int(input('How many data points?'))         

#The grid (especially x-grid) is training data to help MARS work
x_grid = pd.Series(np.linspace(-(size),size,size))
z_grid = pd.Series(np.sin(x_grid) + 0.5*x_grid)

mars_matrix_train = np.zeros((size,size))
mars_matrix_test = np.zeros((size,size))

for row in range(size):
    mars_matrix_train[:,row] = x.apply(lambda a: np.maximum(0.,a-x.iloc[row])).values
    mars_matrix_test[:,row] = x_grid.apply(lambda a: np.maximum(0.,a-x.iloc[row])).values
    
lasso = Lasso(.001)
lasso.fit(mars_matrix_train, z.values.reshape(-1,1))

z_pred = lasso.predict(mars_matrix_test)

plt.figure(figsize = (8,6))
plt.scatter(x,z)
plt.plot(x_grid, z_grid, color = "orange", lw = 4)

plt.plot(x_grid,z_pred, color = "red", lw=3)
plt.legend(["Truth", "Predicted", "Random Draws"])

yeh = int(input('Cycle no.?'))
pred = z_pred[yeh-1]
print('Prediction w/ MARS:', pred)

#MLA Citation
#Sarem, Sarem, and Giridhar. Numbers and Code, 11 Sept. 2018, 
#numbersandcode.com/non-greedy-mars-regression. 