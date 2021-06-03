# -*- coding: utf-8 -*-
"""
Created on Wed May 19 01:37:54 2021

@author: Nicholas
"""

from collections import Counter
import math
import csv
import matplotlib.pyplot as plt

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    for index, example in enumerate(data):
        # This calculates distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[:-1], query)
        
        # This adds that distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # This sorts the distances and indices from smallest to largest
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # This picks the first K points sorted from the collection that are least spaced out
    # Different values of K yield different estimates. One K-value might be the best depending 
    # on how close it is to the real value.
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    return choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i-1] - point2[i-1], 2)
    return math.sqrt(sum_squared_distance)


# Reads Data from CSV File
file_name = str(input('File name:'))
def read_lines():
    with open(file_name + '.csv', 'r') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]
        
dataset = list(read_lines())

print(dataset)

#Creates what needs to be plotted
x = []
z = []
zhatt = []
for point in dataset:
    x.append(point[0])
    z.append(point[2])
 
k = int(input('K-value to use'))
 
for cyc in range(len(x)):
    cycle = []
    cycle.append(cyc)
    zhattt = knn(dataset, cycle , k, distance_fn=euclidean_distance, choice_fn=mean)
    zhatt.append(zhattt)      

#Individual Calculation done here    
cap = float(input('Cycle no.? Ratio no. is same across all data pts.'))
capacitance = []
capacitance.append(cap)

zhat = knn(dataset, capacitance , k, distance_fn=euclidean_distance, choice_fn=mean)

#Last printed value id the predicted # of cycles for the supercapacitor
print('Prediction w/ KNN:', zhat)

#Plots the Prediction Points (blue dots) vs Actual Data (green line)
plt.plot(x, z, 'g-')
plt.plot(x, zhatt, 'bo')
plt.legend(["Real Data", "Predicted"])
plt.show()

#Calculates R Squared for Comparing Algorithms
import numpy

corr_matrix = numpy.corrcoef(z, zhatt)
corr = corr_matrix[0,1]
R_sq = corr**2
 
print('R squared value for KNN:' , R_sq)

