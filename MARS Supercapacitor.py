# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:29:03 2021

@author: Nicholas
"""

#MARS Implemented with Lasso

import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
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

size = len(x_)         

#The x-grid (especially x-grid) is input training data to help MARS work
x_grid = pd.Series(np.linspace(-size,size,size))

mars_matrix_train = np.zeros((size,size))
mars_matrix_test = np.zeros((size,size))

for row in range(size):
    mars_matrix_train[:,row] = x_grid.apply(lambda a: np.maximum(0.,a-x.iloc[row])).values
    mars_matrix_test[:,row] = x.apply(lambda a: np.maximum(0.,a-x.iloc[row])).values
    
lasso = Lasso(.001)
lasso.fit(mars_matrix_test, z)

z_pred = lasso.predict(mars_matrix_test)

plt.plot(x_, z_, color = "orange")

plt.plot(x_,z_pred,'go')
plt.legend(["Real Data", "Predicted"])

yeh = int(input('Cycle no.?'))
pred = z_pred[yeh-1]
print('Prediction w/ MARS:', pred)

#Calculates R Squared for Comparing Algorithms

corr_matrix = np.corrcoef(z_, z_pred)
corr = corr_matrix[0,1]
R_sq = corr**2
 
print('R squared value for MARS:' , R_sq)

#MLA Citation
#Sarem, Sarem, and Giridhar. Numbers and Code, 11 Sept. 2018, 
#numbersandcode.com/non-greedy-mars-regression. 