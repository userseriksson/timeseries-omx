import streamlit as st
import altair as alt
import pickle
# streamlit run omx.py

#----- Packet ------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf # Import library to access Yahoo finance stock data
#from ta import add_all_ta_features
import fastai.tabular
#import matplotlib.pyplot as plt
import sys
import plotly.graph_objs as go  # Import the graph ojbects
import fastai.tabular

import matplotlib.pyplot as plt
from datetime import timedelta, date

from sklearn.metrics import mean_squared_error # Install error metrics 
from sklearn.linear_model import LinearRegression # Install linear regression model
from sklearn.neural_network import MLPRegressor # Install ANN model 
from sklearn.preprocessing import StandardScaler # to scale for ann

#--------------------------- Functions ---------------------------

#load data and select company
def select_comapny(df,name):
    work_data = df[df['Name']==name]
    work_data = work_data.drop(['Name'], axis=1)
    return work_data
# Create the lags 
def CreateLags(df,lag_size):
  # inputs: dataframe , size of the lag (int)
  # ouptut: dataframe ( with extra lag column), shift size (int)

  # add lag
    shiftdays = lag_size
    shift = -shiftdays
    df['Close_lag'] = df['Close'].shift(shift)
    return df, shift

# Split the testing and training data 
def SplitData(df, train_pct, shift):
  # inputs: dataframe , training_pct (float between 0 and 1), size of the lag (int)
  # ouptut: x train dataframe, y train data frame, x test dataframe, y test dataframe, train data frame, test dataframe

    train_pt = int(len(df)*train_pct)
  
    train = df.iloc[:train_pt,:]
    
    test = df.iloc[train_pt:,:]

    x_train = train.iloc[:shift,1:-1]
    y_train = train['Close_lag'][:shift]
    x_test = test.iloc[:shift,1:-1]
    y_test = test['Close'][:shift]

    return x_train, y_train, x_test, y_test, train, test

# Function to make the plots
def PlotModelResults_Plotly(train, test, pred, ticker, w, h, shift_days,name):
      # inputs: train dataframe, test dataframe, predicted value (list), ticker ('string'), width (int), height (int), shift size (int), name (string)
      # output: None

      # Create lines of the training actual, testing actual, prediction 
    D1 = go.Scatter(x=train.index,y=train['Close'],name = 'Train Actual') # Training actuals
    D2 = go.Scatter(x=test.index[:shift],y=test['Close'],name = 'Test Actual') # Testing actuals
    D3 = go.Scatter(x=test.index[:shift],y=pred,name = 'Our Prediction') # Testing predction

      # Combine in an object  
    line = {'data': [D1,D2,D3],
              'layout': {
                  'xaxis' :{'title': 'Date'},
                  'yaxis' :{'title': '$'},
                  'title' : name + ' - ' + tickerSymbol + ' - ' + str(shift_days)
              }}
      # Send object to a figure 
    fig = go.Figure(line)

      # Show figure
    fig.show()

# Regreesion Function

def LinearRegression_fnc(x_train,y_train, x_test, y_test):
      #inputs: x train data, y train data, x test data, y test data (all dataframe's)
      # output: the predicted values for the test data (list)

    lr = LinearRegression()
    lr.fit(x_train,y_train)
    lr_pred = lr.predict(x_test)
    lr_MSE = mean_squared_error(y_test, lr_pred)
    lr_R2 = lr.score(x_test, y_test)
    print('Linear Regression R2: {}'.format(lr_R2))
    print('Linear Regression MSE: {}'.format(lr_MSE))

    return lr_pred

# ANN Function 

def ANN_func(x_train,y_train, x_test, y_test):

      # Scaling data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    MLP = MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes = (100,), activation = 'identity',learning_rate = 'adaptive').fit(x_train_scaled, y_train)
    MLP_pred = MLP.predict(x_test_scaled)
    MLP_MSE = mean_squared_error(y_test, MLP_pred)
    MLP_R2 = MLP.score(x_test_scaled, y_test)

    print('Muli-layer Perceptron R2 Test: {}'.format(MLP_R2))
    print('Multi-layer Perceptron MSE: {}'.format(MLP_MSE))

    return MLP_pred

def LSTM_func(x_train,y_train, x_test, y_test):

      # Scaling data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    MLP = MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes = (1000,), activation = 'identity',learning_rate = 'adaptive').fit(x_train_scaled, y_train)
    MLP_pred = MLP.predict(x_test_scaled)
    MLP_MSE = mean_squared_error(y_test, MLP_pred)
    MLP_R2 = MLP.score(x_test_scaled, y_test)

    print('Muli-layer Perceptron R2 Test: {}'.format(MLP_R2))
    print('Multi-layer Perceptron MSE: {}'.format(MLP_MSE))

    return MLP_pred

def CalcProfit(test_df,pred,j):
    pd.set_option('mode.chained_assignment', None)
    test_df['pred'] = np.nan
    test_df['pred'].iloc[:-j] = pred
    test_df['change'] = test_df['Close_lag'] - test_df['Close'] 
    test_df['change_pred'] = test_df['pred'] - test_df['Close'] 
    test_df['MadeMoney'] = np.where(test_df['change_pred']/test_df['change'] > 0, 1, -1) 
    test_df['profit'] = np.abs(test['change']) * test_df['MadeMoney']
    print(test_df[['Close','Close_lag','pred', 'change', 'MadeMoney','profit']].head())
    profit_dollars = test['profit'].sum()
    print('Would have made: $ ' + str(round(profit_dollars,1)))
    profit_days = len(test_df[test_df['MadeMoney'] == 1])
    print('Percentage of good trading days: ' + str( round(profit_days/(len(test_df)-j),2))     )

    return test_df, profit_dollars


#----- display code --------

#read in data
df_work = pd.read_csv('/Users/joeriksson/Desktop/python_data/stock_swe_2021401.csv', sep = ',')
#print("Hur ser data frame ut",df_work.head())


st.title('Select the stock you wich to predict')
st.write(pd.DataFrame(df_work))

mylist = list(set(df_work['Name']))
option = st.selectbox('Select stock to predict?',(mylist))
st.write('You selected:', option)

#st.title("You have select the stock ", option)
df=select_comapny(df_work, option)
st.write(pd.DataFrame(df))

df= df[['Date','Close']]
df['year_month'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
#df = df[['year_month','Close']]
st.write(pd.DataFrame(df))
df=df.groupby(by="year_month").sum()

df = df.set_index('year_month')


st.line_chart(df)
