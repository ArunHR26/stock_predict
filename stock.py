import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Prediction")

stocks = ("LLOYDSENGG.NS", "NHPC.NS", "UPL.NS", "GENSOL.NS", "JIOFIN.NS")
selected_stock = st.selectbox("Select stock", stocks)
n_days = st.slider("Number of days to predict", 30, 365)

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

def train_model(data):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    df = train_data[['Date', 'Close']]
    df = df.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet(daily_seasonality=True)
    m.fit(df)

    return m, df, test_data

def evaluate_model(m, df, test_data):
    # Evaluate the model
    test_df = test_data[['Date', 'Close']]
    test_df = test_df.rename(columns={"Date": "ds", "Close": "y"})
    forecast_test = m.predict(test_df)

    mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
    accuracy = 1 - mae / test_df['y'].mean()

    return accuracy, test_df, forecast_test

m, df, test_data = train_model(data)
accuracy, test_df, forecast_test = evaluate_model(m, df, test_data)

while accuracy < 0.75:
    m, df, test_data = train_model(data)
    accuracy, test_df, forecast_test = evaluate_model(m, df, test_data)

st.write(f"Model accuracy on test data: {accuracy * 100:.2f}%")

# Generate forecast
future_dates = pd.date_range(start=data['Date'].max(), periods=n_days + 1)[1:]
future = pd.DataFrame({'ds': future_dates})
forecast = m.predict(future)

st.subheader(f"{selected_stock} Forecasting Data")
st.write(forecast.tail(n_days))

st.write('Forecast Data')
fig = plot_plotly(m, forecast)

# Add test data to the plot with black color and non-bold
fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], mode='markers', marker=dict(color='black', symbol='circle', size=4), line=dict(color='black', width=1), name='Test Data'))

st.plotly_chart(fig)

st.write('Forecast Components')
fig = m.plot_components(forecast)
st.write(fig)
