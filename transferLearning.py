# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:37:29 2021

@author: SMedepalli



TL model using specified time series, specified dict, specified LSTM model 
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import itertools
import csv
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
###placeholding dict just to write model


def build_LSTM(input_size, hidden_units, dropout, learning_rate):
    h = hidden_units
    
    model = Sequential()
    
    if isinstance(h,list):
    
        model.add(LSTM(h[0], 
                   batch_input_shape=(1,input_size, 1), 
                   return_sequences=True, 
                   stateful=True))
                  
        if dropout:
            model.add(Dropout(rate=0.5))

        if len(h) > 2:
            #removing 1st and last units
            for index, units in enumerate(h[1:-1]):  
                model.add(LSTM(units, 
                               batch_input_shape=(1,h[index], 1), 
                               return_sequences=True, 
                               stateful=True)) 
                if dropout:
                    model.add(Dropout(rate=0.5))

        model.add(LSTM(h[-1], 
                       batch_input_shape=(1,h[-2], 1), 
                       return_sequences=False, 
                       stateful=True))
        if dropout:
            model.add(Dropout(rate=0.5))
    else:
        model.add(LSTM(h, 
                   batch_input_shape=(1,input_size, 1), 
                   return_sequences=False, 
                   stateful=True)) 
        if dropout:
            model.add(Dropout(rate=0.5))
        
    
    model.add(Dense(1))
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    return model

def predict_ahead(model,X_test,n_ahead):
    predictions = np.zeros(n_ahead)
    predictions[0] = model.predict(X_test,batch_size = 1)
    
    if n_ahead > 1:
        for i in range(1,n_ahead):
            x_new = np.append(X_test[0][1:],predictions[i-1])
            X_test = x_new.reshape(1,x_new.shape[0],1)
            predictions[i] = model.predict(X_test,batch_size = 1)
    return predictions

def preprocessing(series):
    series = np.array(series)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1,1))
    scaled_series = scaled.reshape((len(series),))  
    return scaled_series, scaler

def getSeries(data,p):

    n = data.shape[0]
    n_train = int(n * p) 
    n_test = n - n_train

    x = np.arange(n)
        
    index_train = x[:n_train]
    index_test = x[n_train:]
    
    series = data[index_train]

    y_test = data[index_test]
    return series, y_test, n_test


def getInputOutput(series, input_size):
    
    series = np.array(series)
    xlen = len(series)
    xrows = xlen - input_size
    
    X_train, y_train = [], []
    
    #print(xrows)
    
    for i in range(xrows-1):
        j = i + input_size
        a = series[i:j, np.newaxis]
        
        #print(series[j])
        X_train.append(a)
        y_train.append(series[j])
    
    X_train,y_train = np.array(X_train), np.array(y_train)
    X_test = series[xrows:].reshape(1,input_size,1)
    
    
    return X_train, y_train, X_test

def inverse_transform(series,scaler):
    return scaler.inverse_transform(series.reshape(-1,1))

def graphTL(series,predictions,actual,actualARR,firstVAL,title):
    """
    every50=np.array([])
    FVARR=np.array([])
    x=1
    for i in range(len(predictions)):
       if x==50:
           every50.extend(predictions[i])
           x=1
       else:
           x+=1
           """
    
    
    plt.figure(figsize=(8,4))
    plt.title(title)
    plt.yscale("log")
    
    if isinstance(series,list):
        train_index = np.arange(len(series[0]))
        test_index = len(series[0]) + np.arange(len(actual))
        
        plt.plot(train_index,series[0], label = 'general')
        
    else:
        train_index = np.arange(len(series))
        test_index = len(series) + np.arange(len(actual))        
        plt.plot(train_index,series,label = 'training')

    if len(predictions) > 4:
        plt.plot(test_index,predictions,label = 'predictions',color='g')
        plt.plot(test_index,actualARR,label = 'actual',color='orange')
        #plt.scatter(every50,label = 'every 50',color='r')
        
    else:
        plt.scatter(test_index,predictions,label = 'prediction',color='g') 
        plt.scatter(test_index,actualARR,label = 'actual',color='orange')
        #plt.scatter(every50,label = 'every 50',color='r')
   ##carbon based for range        
    
    plt.xlabel('Cycles')
    plt.ylabel('Capacitance')
    
    plt.legend(loc='upper left')
    plt.savefig('{}_{}.png'.format(title,len(series)))
    plt.show()    
    
    """
    every50=np.array([])
    FVARR=np.array([])
    x=1
    for i in range(len(predictions)):
       if x==50:
           every50.extend(predictions[i])
           x=1
       else:
           x+=1

    for i in range(len(predictions)):
        FVARR.np.append(firstVAL)
    
    
    fig, ax=plt.subplots()
    ax.scatter(FVARR, every50)
    ax.plot([FVARR.min(), FVARR.max()], [FVARR.min(), FVARR.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    """
    
    
    
    """
    fig, ax=plt.subplots()
    ax.scatter(actualARR, predictions)
    ax.plot([actualARR.min(), actualARR.max()], [actualARR.min(), actualARR.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    """

def MSE(y,y_bar):
    summation=0
    n=len(y)
    for i in range(0,n):
        diff=y[i]-y_bar[i]
        sqdiff=diff**2
        summation=summation+sqdiff
    MSE=summation/n
    return MSE

def FitForecast(X_train, y_train, X_test, n_ahead, input_size,
                hidden_units, dropout, val_split, learning_rate, 
                epochs, trained_model):
    
    model = build_LSTM(input_size,hidden_units,dropout, learning_rate)
    
    if trained_model is not None:
        model.set_weights(weights = trained_model.get_weights())        
    
    
    history = model.fit(x=X_train, y=y_train, 
                batch_size=1, epochs=epochs, 
                verbose=1, validation_split=val_split,
                shuffle=False)

    predictions = predict_ahead(model,X_test,n_ahead)
    return model, predictions, history


def DielecConst(series, predictions, actual):
    
    Carr=np.array([])
    for i in range (len(actual)):
        Carr=np.append(Carr, actual[i])
    
    predCarr=np.array([])
    for i in range(len(predictions)):
        predCarr=np.append(predCarr, predictions[i])
        
    ##Eo: 0.0000088541878128 mF/cm, no calc on me square it to make it mF/cm^2
    ##Need 2 get A/d
    d=2*math.sqrt(4/math.pi)
    Eo=0.0000088541878128*0.0000088541878128
    Karr=np.array([])
    for i in range (len(actual)):
        dc=(d*actual[i])/(Eo*4)
        Karr=np.append(dc, Karr)
        
    predKarr=np.array([])
    for i in range (len(predictions)):
        dc=(d*predictions[i])/(4*Eo)
        predKarr=np.append(dc, predKarr)
        
    cycles=np.array([])
    count=0
    for i in range (len(Karr)):
        cycles=np.append(count, cycles)
        count+=1
        
    predCycles=np.array([]) 
    count=0
    for i in range (len(Karr)):
        predCycles=np.append(count, predCycles)
        count+=1
        
    
    #print(Karr) 
    q3,q1=np.percentile(Karr, [75,25])
   # print(q3-q1)
        
    ConstraintRangeArr=np.array([])
    for i in range (len(Karr)):
        if((Karr[i]<q3) & (Karr[i]>q1)):
            ConstraintRangeArr=np.append(Karr[i], ConstraintRangeArr)
            
    #print(ConstraintRangeArr)
    
    back2CapArr=np.array([])
    for i in range(len(ConstraintRangeArr)):
        c=Eo*ConstraintRangeArr[i]*(4/d)
        back2CapArr=np.append(c,back2CapArr)
        
    print("KMSE",MSE(ConstraintRangeArr,back2CapArr))
    
    f=open("ConstrainedCapp.csv", "a")
    for i in range(len(back2CapArr)):
        f.write(str(back2CapArr[i]))
        f.write("\n")
    f.close()
    
           
    
    plt.figure(figsize=(8,4))
    plt.title("Dielectric Constant(k)")
    plt.yscale("log")
    """
    print("test1")
    plt.plot(cycles, Karr,label="dielectric constants", color='r')
    print("test2")
    plt.plot(ConstraintRangeArr, label='pred dielectric constants', color='b')
    print("test3")
    """
    plt.plot(back2CapArr,label='Predicted Capacitance Bounded by K Range', color='b')
    plt.plot(ConstraintRangeArr, label='Actual Capacitance Bounded by K Range', color='r')
    plt.xlabel('Cycles')
    plt.ylabel('Dielectric Constant')   
    plt.legend(loc='upper left')
    plt.show()
    
        
###k needs to be in that range

####Predict the reversed values move this function to a new file


###Every 50##########################################################################
def every50(series,predictions,actual):
    firstVal=actual[0]
    count=0;
    for i in range (len(actual)):
        if(count%50==0):
            if(actual[i]>=firstVal):
                actual[i]=firstVal
        count+=1
                
                
    f=open("Every50Cap.csv", "a")
    for i in range(len(actual)):
        f.write(str(actual[i]))
        f.write("\n")
    f.close()
###end every50 const####################################################################
    
####100%#####################################################
def hunCon(series,actual):
    f=open("hunConData.csv", "a")
    for i in range(len(actual)):
        if(actual[i]<=actual[0]):
            f.write(str(actual[i]))
            f.write("\n")
    f.close()
######################################################################
    

def perfTL(time_series,input_size,hidden_units,dropout,learning_rate,n_ahead,val_split,epochs,verbose,plot,model):    
    
    #count=0
    actualARR=np.array([])
    
    capARR=np.array([])
    with open('actualvalues1.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            capARR= np.append(capARR, float(row[0]))
            
           
                
    with open('actualForGraphing.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                actualARR= np.append(actualARR, float(row[0]))
                
            
    fullCapArr=np.array([])
    with open('realtest1.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            fullCapARR= np.append(fullCapArr, float(row[0]))
    
    val_split = 0
    
    scaled_series, scaler = preprocessing(time_series)
    series, y_test, n_test = getSeries(scaled_series,0.8)
    X_train,y_train,X_test = getInputOutput(series,input_size)

    # show only n_ahead number of actual values
    #print(y_test[np.arange(n_ahead)])
    y_test = np.arange(n_ahead)
    #print(y_test)
    
    print('*** Fitting a model without knowledge transfer ***')
    model_noTransfer, predictions_noTransfer, _ = FitForecast(X_train,y_train,
                                                             X_test,n_ahead,
                                               input_size,hidden_units,
                                                             dropout,val_split,
                                                             learning_rate,
                                               epochs,
                                                             trained_model=None)
    
    print('\n')
    print('*** Fitting a model with knowledge transfer ***')
    model_withTransfer, predictions_withTransfer, _ = FitForecast(X_train,y_train,
                                                                 X_test,n_ahead,
                                                                 input_size,hidden_units,
                                                                 dropout,val_split,
                                                                 learning_rate,
                                                                 epochs,
                                                                 trained_model=model)
    

    
    # rescaling,,,,train on moving average line
    print("test1")
    series = inverse_transform(series, scaler)
    print("test2")
    y_test = inverse_transform(y_test, scaler)
    print("test3")
    predictions_noTransfer = inverse_transform(predictions_noTransfer, scaler)
    predictions_withTransfer = inverse_transform(predictions_withTransfer, scaler)
    
    #print(predictions_noTransfer)
    #print(predictions_withTransfer)
    print("test4")
    #mse1 = mean_squared_error(y_true=actualARR,y_pred=predictions_noTransfer)
    print('test5')
    #mse2 = mean_squared_error(y_true=actualARR,y_pred=predictions_withTransfer)
    print(len(actualARR))
    print(len(predictions_noTransfer))
    
    mse1= MSE(actualARR,predictions_noTransfer)
    mse2=MSE(actualARR,predictions_withTransfer)
    print('MSE without TL: ',mse1)
    print('MSE with TL: ',mse2)
    
    #print(y_test)
    #DielecConst(time_series, predictions_noTransfer, capARR)
   # every50(time_series,predictions_noTransfer,capARR)
    #hunCon(time_series,capARR)
    
    firstVAL=capARR[0]
    
    """
    every50=np.array([])
    FVARR=np.array([])
    x=1
    fifcount=0
    for i in range(len(predictions_noTransfer)):
       if x==50:
           every50=np.append(every50, predictions_noTransfer[i])
           x=1
           fifcount+=1
       else:
           x+=1

    for i in range(fifcount):
        FVARR=np.append(FVARR, firstVAL)
    
    plt.figure(figsize=(8,4))
    plt.title("Without TL")
    plt.plot(predictions_noTransfer,label = 'predictions',color='b')
    plt.plot(predictions_withTransfer,label='predictions with TL',color='g')
    plt.plot(FVARR,label='first value',linestyle='dashed',color='r')
    
    """
    
    
    
    """
    fig, ax=plt.subplots()
    ax.scatter(FVARR, every50)
    ax.plot([FVARR.min(), FVARR.max()], [FVARR.min(), FVARR.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    
    every50.clear()
    x=1
    for i in range(len(predictions_noTransfer)):
       if x==50:
           every50=np.append(every50, predictions_noTransfer[i])
           x=1
       else:
           x+=1
    
    fig, ax=plt.subplots()
    ax.scatter(FVARR, every50)
    ax.plot([FVARR.min(), FVARR.max()], [FVARR.min(), FVARR.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
        
    graphTL(series,predictions_noTransfer,y_test,actualARR,firstVAL,title='Without Transfer')
    graphTL(series,predictions_withTransfer,y_test,actualARR,firstVAL,title='With Transfer')
    
    """
    
    graphTL(series,predictions_noTransfer,y_test,actualARR,firstVAL,title='Without Transfer')
    graphTL(series,predictions_withTransfer,y_test,actualARR,firstVAL,title='With Transfer')
    
    
    
    
############ everything below will be implementation based on our data    
    
input_size=1000
hidden_units=[100,50]
dropout=False
learning_rate=4e-5
n_ahead=399
val_split=0.2
epochs=1
verbose=True
plot=False

capARR=[]
with open('realtest1.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        capARR.append(float(row[0])) 

##print(len(capARR))

series=pd.Series(capARR)
##print(series)

#print(len(np.array(series))-input_size)


model=build_LSTM(input_size,hidden_units,dropout,learning_rate)
perfTL(series,input_size,hidden_units,dropout,learning_rate,n_ahead,val_split,epochs,verbose,plot,model)   






"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from google.colab import files
trainData = files.upload()
actualData=files.upload()
#from prophet.plot import plot_plotly, plot_components_plotly
import io
actualDataDF=pd.read_csv(io.BytesIO(actualData['actualData(prophet).csv']))
df=pd.read_csv(io.BytesIO(trainData['prophetDATA.csv']))
df.head()
df.columns=['ds','y']
actualDataDF.columns=['y']


m=Prophet()
m=Prophet(daily_seasonality=True)
df['floor']=0
future['floor']=0
df['cap']=3
future['cap']=3
#m=Prophet(growth='logistic')
m.fit(df)


future=m.make_future_dataframe(periods=1)
future.tail()

forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
print(forecast)
print(forecast.loc[:,'yhat'])
print(actualDataDF)

actualIndex=np.arange(1000,2000)
predIndex=np.arange(1000,2000)


plt.figure(figsize=(8,4))
plt.yscale("log")
plt.plot(df.loc[:,'y'],label='training')
plt.plot(actualIndex,actualDataDF.loc[:,'y'],label='actual',color='orange')
plt.plot(predIndex,forecast.loc[:,'yhat'],label='predictions',color='g')

plt.xlabel('Cycles')
plt.ylabel('Specific Capacitance')    
plt.legend(loc='upper left')

#fig1=m.plot(forecast)
#fig1.yscale("log")
#fig2=m.plot_components(forecast)
"""
