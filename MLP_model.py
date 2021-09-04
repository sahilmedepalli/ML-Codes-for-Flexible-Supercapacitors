import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
#import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
#from sklearn import tree

#------ DATA FRAMES ------#
df1 = pd.read_csv('LS_12(0.1).csv')
df2 = pd.read_csv('LS_12(1).csv')
df3 = pd.read_csv('LS_24(0.1).csv')
df4 = pd.read_csv('LS_24(1).csv')

megaLS = pd.concat([df1, df2, df3])

df5 = pd.read_csv('DAL_12(0.1).csv')
df6 = pd.read_csv('DAL_12(1).csv')
df7 = pd.read_csv('DAL_24(0.1).csv')
df8 = pd.read_csv('DAL_24(1).csv')

megaDAL = pd.concat([df5, df6, df7], ignore_index=True)

df9 = pd.read_csv('LIG_12(0.1).csv')
df10 = pd.read_csv('LIG_24(0.1).csv')
#-------------------------#

#Handle NaN values
#TODO: Experiment with log scale with values that return NaN
df5.ffill(inplace=True)
df6.ffill(inplace=True)
df7.ffill(inplace=True)
df8.ffill(inplace=True)
df9.ffill(inplace=True)
df10.ffill(inplace=True)
megaDAL.ffill(inplace=True)

train = df4.iloc[:101]
test = df4.iloc[101:]
print(train.shape, test.shape)

model = MLPRegressor(solver='lbfgs', warm_start=True)
#solver='lbfgs', warm_start=True

model.fit(train, train.target)
pred = model.predict(test)

score = model.score(train, train.target)
print('R^2: ', score)

mse = mean_squared_error(test.target, pred)
print('MSE: ', mse)

fig1 = plt.figure(num=1, dpi=300, facecolor='w', edgecolor='k')
plt.plot(test.target, pred, 'ro', markersize=3)
plt.plot(test.target, test.target, color='black')
plt.xlabel(r'Actual Specific Capacitance (uF/cm$^2$)')
plt.ylabel(r'Predicted Specific Capacitance (uF/cm$^2$)')

fig2 = plt.figure(num=2, dpi=300, facecolor='w', edgecolor='k')
plt.plot(train.index, train.target, 'bo', markersize=2, label='Train')
#plt.plot(test.index, test.target, label='Test')
#plt.plot(test.index, pred, linewidth=.5, label='Predictions')
#plt.legend(['Train', 'Test', 'Predicitons'], loc='lower right')
plt.ylabel('Specific Capacitance')
plt.xlabel('Cycle No.')

fig3 = plt.figure(num=3, dpi=300, facecolor='w', edgecolor='k')
plt.plot(train.index, train.target, label='Training')
plt.plot(test.index, test.target, label='Test')
plt.plot(test.index, pred, linewidth=.5, label='Predictions')
plt.legend(['Training', 'Test', 'Predicitons'], loc='lower right')
plt.ylabel('Specific Capacitance (uF/cm$^2$)')
plt.xlabel('Cycle No.')
plt.title('Specific Capacitance - LS_24h(1:10)')