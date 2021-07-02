# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 00:41:57 2021

@author: Nicholas
"""

#MARS Implemented with Lasso

import csv
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

#Creates Training and Testing Data Split
X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=50, random_state=0) 
    
lasso = Lasso(.001)
lasso.fit(X_train, z_train)

z_pred = lasso.predict(X_test)

plt.plot(z_, z_, color = "orange")

plt.plot(z_test,z_pred,'go')
plt.legend(["Real Data", "Predicted"])

#Calculates MSE for Comparing Algorithms

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(z_test, z_pred)
print('MSE:', MSE)


#MLA Citation
#Sarem, Sarem, and Giridhar. Numbers and Code, 11 Sept. 2018, 
#numbersandcode.com/non-greedy-mars-regression. 