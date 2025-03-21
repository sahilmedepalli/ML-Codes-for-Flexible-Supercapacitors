# -*- coding: utf-8 -*-
"""
Created on Sat May  8 13:24:17 2021

@author: Nicholas
"""

from collections import Counter
import math

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

    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

    
#Reads Data from Supercapacitor Data
import csv

file_name = str(input('Name of CSV file:'))
with open(file_name + '.csv', 'r') as datfile:

    capacitance_vs_cycleno = [[float(x), float(y)] for x, y in csv.reader(datfile, delimiter= ',')]
    
print(capacitance_vs_cycleno)    
    
cap = float(input('Cycle no.?'))
capacitance = []
capacitance.append(cap)
k = int(input('K-value to use'))

#Last printed value id the predicted # of cycles for the supercapacitor
print(knn(capacitance_vs_cycleno, capacitance , k, distance_fn=euclidean_distance, choice_fn=mean))


#MLA Citation of Source of Code Idea:
# Harrison, Onel. “Machine Learning Basics with the K-Nearest Neighbors Algorithm.” 
# Medium, Towards Data Science, 14 July 2019, towardsdatascience.com/machine-learning-
# basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761.
