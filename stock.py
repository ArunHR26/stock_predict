import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pandas.tseries.offsets import CustomBusinessDay
import requests

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Prediction")

stocks = ("LLOYDSENGG.NS", "NHPC.NS", "UPL.NS", "GENSOL.NS", "JIOFIN.NS", "HDFCBANK.NS", "ADANIGREEN.NS")
selected_stock = st.selectbox("Select stock", stocks)

# Adjust the number of days to predict
n_days = st.slider("Number of days to predict", 7, 365)

# Function to load data with volume
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Download Indian holidays data from the internet
def download_indian_holidays():
    indian_holidays_url = "https://www.officeholidays.com/ics/india"
    response = requests.get(indian_holidays_url)
    ical_data = response.text

    # Parse the downloaded data to extract holiday dates
    holidays = []
    for line in ical_data.split("\n"):
        if line.startswith("DTSTART"):
            holiday_date = line.split(":")[1]
            holidays.append(pd.to_datetime(holiday_date))
    return holidays

# Excluding Indian holidays and weekends
indian_holidays = download_indian_holidays()
indian_bday = CustomBusinessDay(holidays=indian_holidays)

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

# Modify train_model function to include volume
def train_model(data, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale):
    train_size = int(len(data) * 0.999)
    train_data, test_data = data[:train_size], data[train_size:]

    df = train_data[['Date', 'Close', 'Volume']]  # Include Volume
    df = df.rename(columns={"Date": "ds", "Close": "y", "Volume": "volume"})  # Rename columns

    m = Prophet(
        daily_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale
    )

    m.add_regressor('volume')  # Add volume as regressor
    m.fit(df)

    return m, df, test_data

# Modify evaluate_model function
def evaluate_model(m, df, test_data):
    test_df = test_data[['Date', 'Close', 'Volume']]  # Include Volume
    test_df = test_df.rename(columns={"Date": "ds", "Close": "y", "Volume": "volume"})  # Rename columns
    forecast_test = m.predict(test_df)

    mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
    accuracy = 1 - mae / test_df['y'].mean()

    return accuracy, test_df, forecast_test

best_accuracy = 0
best_model = None
best_df = None
best_test_data = None

# Hyperparameter tuning
for changepoint_prior_scale in [0.001, 0.01, 0.1, 0.5]:
    for seasonality_prior_scale in [0.01, 0.1, 1.0, 10.0]:
        for holidays_prior_scale in [0.01, 0.1, 1.0, 10.0]:
            m, df, test_data = train_model(data, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale)
            accuracy, test_df, _ = evaluate_model(m, df, test_data)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = m
                best_df = df
                best_test_data = test_data

st.write(f"Best model accuracy on test data: {best_accuracy * 100:.2f}%")

# Generate forecast
future_dates = pd.date_range(start=data['Date'].max(), periods=n_days, freq=indian_bday)
future_dates = pd.Index([date.today()] + future_dates.tolist())  # Include current date explicitly
future = pd.DataFrame({'ds': future_dates})
future['volume'] = best_df['volume'].iloc[-1]  # Include volume data for forecast
forecast = best_model.predict(future)

st.subheader(f"{selected_stock} Forecasting Data")
st.write(forecast)

st.write('Forecast Data')
fig = plot_plotly(best_model, forecast)
fig.add_trace(go.Scatter(x=test_df['ds'], y=test_df['y'], mode='markers'))
st.plotly_chart(fig)

st.write('Forecast Components')
fig = best_model.plot_components(forecast)
st.write(fig)
