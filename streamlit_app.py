import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import joblib as jlb
import streamlit as st

st.header('SBIN Highest Price Prediction using ARIMA model')

def init_app():
  days = st.slider(label = 'Enter Days', max_value = 10, min_value = 1, step = 1)
  make_response(days)

def make_response(days):
  days_array = list(range(1, days+1))
  model = jlb.load('sbin-high.pk1')
  forecasts = model.forecast(steps = days)
  df = pd.DataFrame({ 'Days': days_array, 'Predictions': forecasts})
  st.subheader('Predictions')
  st.table(df)
  st.subheader('Days vs. Price Trend')
  fig = plt.figure()
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.scatter(days_array, forecasts)
  st.pyplot(fig)
  
init_app()
