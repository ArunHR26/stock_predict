import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import lightgbm as lgb
import matplotlib.pyplot as plt

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Prediction")

stocks = ("LLOYDSENGG.NS", "NHPC.NS", "UPL.NS", "GENSOL.NS", "JIOFIN.NS", "HDFCBANK.NS", "ADANIGREEN.NS")
selected_stock = st.selectbox("Select stock", stocks)
n_days = st.slider("Number of days to predict", 7, 365)

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

# Feature engineering
data['Date'] = pd.to_datetime(data['Date'])
data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
data['day_of_month'] = data['Date'].dt.day

# Model training
def train_model(data):
    train_size = int(len(data) * 0.99)
    train_data, test_data = data[:train_size], data[train_size:]

    lgb_train = lgb.Dataset(train_data[['day_of_week', 'month', 'day_of_month']], label=train_data['Close'])
    lgb_eval = lgb.Dataset(test_data[['day_of_week', 'month', 'day_of_month']], label=test_data['Close'], reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'early_stopping_round': 5  # Early stopping rounds
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_eval], verbose_eval=False)

    # Calculate accuracy
    mse = gbm.best_score['valid_1']['l2']
    rmse = np.sqrt(mse)
    accuracy = 100 - rmse
    st.subheader(f"Model Accuracy: {accuracy:.2f}%")

    return gbm, test_data

gbm, test_data = train_model(data)

st.subheader("Forecasting")

# Generate forecast
future_dates = pd.date_range(start=data['Date'].max(), periods=n_days + 1)[1:]
future_data = pd.DataFrame({'Date': future_dates})
future_data['day_of_week'] = future_data['Date'].dt.dayofweek
future_data['month'] = future_data['Date'].dt.month
future_data['day_of_month'] = future_data['Date'].dt.day

forecast = gbm.predict(future_data[['day_of_week', 'month', 'day_of_month']], num_iteration=gbm.best_iteration)

# Plotting
st.subheader(f"{selected_stock} Forecasting Data")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical Data', mode='lines'))
fig.add_trace(go.Scatter(x=future_data['Date'], y=forecast, name='Forecast', mode='lines'))
st.plotly_chart(fig)

# Display tabular view
st.subheader("Forecasting Data")
forecast_df = pd.DataFrame({'Date': future_data['Date'], 'Forecast': forecast})
st.write(forecast_df)

# Plot feature importance
st.subheader("Feature Importance")
importance_fig, ax = plt.subplots()
lgb.plot_importance(gbm, ax=ax, height=0.5)
st.pyplot(importance_fig)
