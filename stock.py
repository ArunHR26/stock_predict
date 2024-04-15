import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pandas.tseries.offsets import CustomBusinessDay
import lightgbm as lgb

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Prediction")

stocks = ("LLOYDSENGG.NS", "NHPC.NS", "UPL.NS", "GENSOL.NS", "JIOFIN.NS", "HDFCBANK.NS", "ADANIGREEN.NS")
selected_stock = st.selectbox("Select stock", stocks)

# Adjust the number of days to predict
n_days = st.slider("Number of days to predict", 4, 365)

# Function to load data
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

# Prepare data for LightGBM
data['Date'] = pd.to_datetime(data['Date'])
data['day_of_week'] = data['Date'].dt.dayofweek
data['quarter'] = data['Date'].dt.quarter
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['day_of_year'] = data['Date'].dt.dayofyear
data['day_of_month'] = data['Date'].dt.day
data['week_of_year'] = data['Date'].dt.isocalendar().week

train_size = int(len(data) * 0.99)
train_data, test_data = data[:train_size], data[train_size:]

train_X = train_data.drop(['Date', 'Close'], axis=1)
train_y = train_data['Close']
test_X = test_data.drop(['Date', 'Close'], axis=1)
test_y = test_data['Close']

# Train LightGBM model
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l1', 'l2'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

st.text("Training LightGBM model...")
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=100)

# Generate forecast
future_dates = pd.date_range(start=data['Date'].max(), periods=n_days, freq=indian_bday)
future_dates = pd.Index([date.today()] + future_dates.tolist())  # Include current date explicitly
future_data = pd.DataFrame({'Date': future_dates})
future_data['day_of_week'] = future_data['Date'].dt.dayofweek
future_data['quarter'] = future_data['Date'].dt.quarter
future_data['month'] = future_data['Date'].dt.month
future_data['year'] = future_data['Date'].dt.year
future_data['day_of_year'] = future_data['Date'].dt.dayofyear
future_data['day_of_month'] = future_data['Date'].dt.day
future_data['week_of_year'] = future_data['Date'].dt.isocalendar().week

forecast = gbm.predict(future_data.drop('Date', axis=1))

st.subheader(f"{selected_stock} Forecasting Data")
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
st.write(forecast_df)

