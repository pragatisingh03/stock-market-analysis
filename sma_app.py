import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained model
model = load_model('analysis_model.h5')

# Streamlit app configuration
st.title('Stock Market Analysis')

# User input for stock ticker and date range
stock = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
start = st.sidebar.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
end = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-12-31'))
future_days = st.sidebar.number_input('Enter number of days for future prediction', min_value=1, max_value=30, value=10)

# Fetch data from Yahoo Finance
data = yf.download(stock, start=start, end=end)

# Plot historical closing prices
fig1 = px.line(data, x=data.index, y=data['Close'], title=f'{stock} Closing Prices')
st.plotly_chart(fig1)

# Sub Tabs for Pricing Data, Visualizations, and News
pricing_data, graphs, future_prediction = st.tabs(["Pricing Data", "Visualizations", "Future Prediction"])

# Pricing Data tab
with pricing_data:
    st.header('Price Movements')
    st.write(f"Data from {start} to {end}")
    data['%Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data.dropna(inplace=True)
    st.write(data)
    annual_return = data['%Change'].mean() * 252 * 100
    st.write(f'Annual return: {annual_return:.2f}%')
    stdev = np.std(data['%Change']) * np.sqrt(252)
    st.write(f'Standard deviation: {stdev * 100:.2f}%')

# Preprocess data for training the model
data = data.reset_index()
data = data.drop(['Date', 'Adj Close'], axis=1)

data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
final_data = pd.concat([past_100_days, data_test], ignore_index=True)
final_data_scaled = scaler.fit_transform(final_data)

# Prepare the data for predictions
x_test = []
y_test = []

for i in range(100, final_data_scaled.shape[0]):
    x_test.append(final_data_scaled[i-100:i])
    y_test.append(final_data_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Visualizations tab
with graphs:
    ma100 = data['Close'].rolling(100).mean()
    ma200 = data['Close'].rolling(200).mean()

    st.subheader('Closing Price vs Time chart with Moving Averages')
    fig2 = plt.figure(figsize=(8,4))
    plt.plot(data['Close'], 'b', label='Closing Price')
    plt.plot(ma100, 'r', label='100-day MA')
    plt.plot(ma200, 'g', label='200-day MA')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    x = []
    y = []

    for i in range(100, final_data_scaled.shape[0]):
        x.append(final_data_scaled[i-100:i])
        y.append(final_data_scaled[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig3 = plt.figure(figsize=(8,4))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig3)

# Future Prediction tab
with future_prediction:
    st.subheader(f'Predicting Future Prices for the next {future_days} days')

    # Use the last 100 days from the original data to predict future prices
    last_100_days = final_data_scaled[-100:]  # Ensure the shape is (100, 1)
    last_100_days = np.array(last_100_days).reshape(1, 100, 1)  # Reshape to (1, 100, 1) for LSTM input

    # Initialize future price prediction
    predicted_prices = []

    for _ in range(future_days):
        # Predict the next day using the model
        next_price_scaled = model.predict(last_100_days)

        # Append the predicted price
        predicted_prices.append(next_price_scaled[0, 0])

        # Update the last 100 days with the new predicted price
        # Drop the first value, add the new predicted value to the end
        next_price_scaled = np.reshape(next_price_scaled, (1, 1, 1))  # Reshape to (1, 1, 1) to match last_100_days
        last_100_days = np.append(last_100_days[:, 1:, :], next_price_scaled, axis=1)

    # Reverse scaling for final predicted prices
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Ensure the table contains predictions for all future days based on user input
    future_dates = pd.date_range(end + pd.Timedelta(days=1), periods=future_days).strftime('%Y-%m-%d')
    future_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])

    # Display future predictions in table 
    st.write(future_df)

    # Plot predicted prices 
    fig3, ax = plt.subplots(figsize=(8, 4))
    ax.plot(future_df.index, future_df['Predicted Price'], 'r', label='Predicted Price')

    # Format the x-axis to avoid overlapping dates
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{stock} Future Prices for next {future_days} days')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    ax.legend()

    # Automatically format the date labels
    fig3.autofmt_xdate()  # Automatically formats dates to prevent overlap

    # Display the graph
    st.pyplot(fig3)
