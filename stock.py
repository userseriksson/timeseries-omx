import streamlit as st
import pandas as pd
import numpy as np
#import chart_studio.plotly as plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go_ob
from datetime import datetime
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

# Extract one stock to predict 
def select_comapny(df,name):
    work_data = df[df['Name']==name]
    work_data = work_data.drop(['Name'], axis=1)
    return work_data

st.title('Stock Forecast App')
data  = pd.read_csv('/Users/joeriksson/Desktop/python_data/stock_swe_2021401.csv',sep = ',',  decimal=",")
stock_name = data['Name'].unique()

#data_set_uniqe = data['shortName'].unique()
option = st.selectbox('Select dataset for prediction',stock_name)
data=select_comapny(data,option)


year = st.slider('90 days interval of prediction:',1,8)
period = year * 90
#DATA_URL =('./HISTORICAL_DATA/3IINFOTECH_data.csv')

#@st.cache

#data_load_state = st.text('Loading data...')
#data 
#data_load_state.text('Loading data... done!')
def Candlestick():
    fig = go.Figure()
    fig = go_ob.Figure(data=[go_ob.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
    fig.layout.update(title_text='Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

Candlestick()

def plot_fig():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='deepskyblue'))
	fig.add_trace(go.Scatter(x=data.Date, y=data['Close'], name="stock_close",line_color='dimgray'))
	fig.layout.update(title_text='Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	return fig

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
	
# plotting the figure of Actual Data
#plot_fig()

# preparing the data for Facebook-Prophet.
data_pred = data[['Date','Close']]
data_pred=data_pred.rename(columns={"Date": "ds", "Close": "y"})

# code for facebook prophet prediction
m = Prophet()
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
m.fit(data_pred)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#plot forecast
fig1 = plot_plotly(m, forecast)
if st.checkbox('Show forecast data'):
    st.subheader('forecast data')
    st.write(forecast)
st.write('Forecasting closing of stock value for '+option+' for a period of: '+str(period)+' Days')
st.plotly_chart(fig1)

#plot component wise forecast
st.write("Component wise forecast")
fig2 = m.plot_components(forecast)
st.write(fig2)
	
