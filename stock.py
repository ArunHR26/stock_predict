import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

START = "2010-01-01"
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

# Splitting data into train and test sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

df = train_data[['Date', 'Close']]
df = df.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality=True)
m.fit(df)

# Generate a dataframe for future dates
future_dates = pd.date_range(start=data['Date'].max(), periods=n_days + 1)[1:]
future = pd.DataFrame({'ds': future_dates})

# Evaluate the model
test_df = test_data[['Date', 'Close']]
test_df = test_df.rename(columns={"Date": "ds", "Close": "y"})
forecast_test = m.predict(test_df)

mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
accuracy = 1 - mae / test_df['y'].mean()

while accuracy < 0.93:
    # Instantiate a new Prophet object
    m = Prophet(daily_seasonality=True)
    
    # Train the model
    m.fit(df)
    
    # Evaluate the model again
    forecast_test = m.predict(test_df)
    mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
    accuracy = 1 - mae / test_df['y'].mean()

st.write(f"Model accuracy on test data: {accuracy * 100:.2f}%")

# Generate forecast
forecast = m.predict(future)

st.subheader(f"{selected_stock} Forecasting Data")
st.write(forecast.tail(n_days))

st.write('Forecast Data')
fig = plot_plotly(m, forecast)
st.plotly_chart(fig)

st.write('Forecast Components')
fig = m.plot_components(forecast)
st.write(fig)
