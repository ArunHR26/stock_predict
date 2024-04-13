import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
import numpy as np
import pandas as pd

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Prediction")

stocks = ("LLOYDSENGG.NS", "NHPC.NS", "UPL.NS", "GENSOL.NS", "JIOFIN.NS")
selected_stock = st.selectbox("Select stock", stocks)
n_days = st.slider("Number of days to predict", 1, 365)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

st.subheader(f"{selected_stock} Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

df = data[['Date', 'Close']]
df = df.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality=True)
m.fit(df)

future = m.make_future_dataframe(periods=n_days)

# Generate a dataframe for future dates
future_dates = pd.date_range(start=data['Date'].max(), periods=n_days + 1)[1:]
future = pd.DataFrame({'ds': future_dates})

forecast = m.predict(future)

st.subheader(f"{selected_stock} Forecasting Data")
st.write(forecast.tail(n_days))

st.write('Forecast Data')
fig = plot_plotly(m, forecast)
st.plotly_chart(fig)

st.write('Forecast Components')
fig = m.plot_components(forecast)
st.write(fig)
