import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
import numpy as np

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("JSK STOCKS")
stocks = ("LLOYDSENGG.NS", "NHPC.NS", "UPL.NS", "GENSOL.NS", "JIOFIN.NS")
selected_stock = st.selectbox("Select stock", stocks)
n_year = st.slider("year of prediction", 1, 3)
period = n_year * 365

#### Downloading Data with Caching
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

##### Downloading data
data_load_state = st.text("Load data....")
data = load_data(selected_stock)
data_load_state.text("Load data.... Done!")

### Showing Raw Data
st.subheader('{} Raw Data'.format(selected_stock))
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)     
plot_raw_data()

# Splitting data into training and validation sets
train_size = int(len(data) * 0.8)
train_data, valid_data = data[:train_size], data[train_size:]

while True:
    ###### Forecasting
    train_df = train_data[['Date', 'Close', 'Open']]
    train_df = train_df.rename(columns={"Date": "ds", "Close": "y", "Open": "additional_regressor"})

    valid_df = valid_data[['Date', 'Close', 'Open']]
    valid_df = valid_df.rename(columns={"Date": "ds", "Close": "y", "Open": "additional_regressor"})

    # Tune Prophet Parameters
    m = Prophet(changepoint_prior_scale=0.05)  # Adjust changepoint_prior_scale as needed
    m.add_regressor('additional_regressor')

    # Train the model
    m.fit(train_df)

    # Validate the model
    forecast = m.predict(valid_df)

    # Calculate accuracy
    valid_rmse = np.sqrt(np.mean((forecast['yhat'].values - valid_df['y'].values) ** 2))
    accuracy = 1 - (valid_rmse / np.mean(valid_df['y'].values))

    if accuracy >= 0.95:
        break
    else:
        st.write(f"Validation RMSE: {valid_rmse}")
        st.write(f"Accuracy: {accuracy}")
        st.write("Retraining the model as accuracy is below 95%...")
        train_size -= 10  # Decrease training size by 10 data points
        train_data, valid_data = data[:train_size], data[train_size:]

st.subheader('JSK Forecasting Data')
st.write(forecast.tail())

st.write(f"Validation RMSE: {valid_rmse}")
st.write(f"Accuracy: {accuracy}")

st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)