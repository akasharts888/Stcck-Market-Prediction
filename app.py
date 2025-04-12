from flask import Flask, render_template,jsonify,request
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
import os
def add_technical_indicators(data):
    # Moving Average (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data


warnings.filterwarnings("ignore", category=UserWarning)
ticker = 'YESBANK.NS'  # For NSE
data = yf.download(ticker, start='2020-01-01', end='2024-11-21')

# Add technical indicators
data = add_technical_indicators(data)


Num_Days = 0

# data = pd.read_csv('Model\Data\Bank_Stock.csv')
testData = data.iloc[:, 4:5]
sc = MinMaxScaler(feature_range=(0, 1))
testData = sc.fit_transform(testData)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

x_train = []
y_train = []
for i in range(60, len(testData)):
    x_train.append(testData[i-60:i, 0])
    y_train.append(testData[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, shuffle=False
)

model = tf.keras.models.load_model("Model/lstm_stock_prediction_model.h5")

# Recompile the model after loading
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Now you can evaluate or make predictions with the model
test_loss, test_mae, test_mse = model.evaluate(x_val, y_val)

app = Flask(__name__)

def update_data(recent_data, next_day_price):
    if isinstance(recent_data, np.ndarray):
        recent_data = pd.DataFrame(recent_data, columns=['Close'])
    # Create a new DataFrame containing the predicted price
    new_row = pd.DataFrame({'Close': [next_day_price]})
    # Concatenate the new row with the recent data
    recent_data = pd.concat([recent_data, new_row], ignore_index=True)

    # Remove the oldest price to maintain a fixed window size
    recent_data = recent_data.tail(60)

    return recent_data
def reshape_data(data):
    # Convert data to numpy array if it's not already
    data_array = np.array(data)

    reshaped_data = data_array.reshape(1, data_array.shape[0], 1).astype(np.float32)

    # Convert data type to float32 for compatibility with TensorFlow
    reshaped_data = reshaped_data.astype(np.float32)

    return reshaped_data

def prepare_data_for_date(num_days_to_predict):
    Num_Days = num_days_to_predict
    predicted_prices = []
    recent_data = data.iloc[:, 4:5].tail(60)
    recent_data = sc.transform(recent_data)
    for _ in range(num_days_to_predict):
        # Reshape recent_data for prediction
        input_data = reshape_data(recent_data)

        # Predict the next day's closing price
        next_day_prediction = model.predict(input_data)

        # Inverse transform the prediction to original scale
        next_day_price = sc.inverse_transform(next_day_prediction)
        # Append the predicted price to the list
        predicted_prices.append(next_day_price[0][0])
        recent_data = update_data(recent_data, next_day_prediction)
        # Update recent_data for the next prediction
    return np.array(predicted_prices)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    try:
        input_days = int(request.form['days'])
        predictions = prepare_data_for_date(input_days)
        current_price = data['Close'].iloc[-1]  # Fetch current price
        last_date = pd.to_datetime(data.index[-1])  # Get the last date from historical data
        next_x_days = pd.date_range(last_date, periods=input_days,freq='B')
        

        # Prepare historical data with SMA
        historical_data = list(zip(
            data.index[-100:],  # Show last 100 rows
            data['Close'].iloc[-100:],
            data['SMA_50'].iloc[-100:],
            data['SMA_200'].iloc[-100:]
        ))

        if len(next_x_days) != len(predictions):
            raise ValueError(f"Mismatch in length: {len(next_x_days)} vs {len(predictions)}")
        
        predicted_data = list(zip(next_x_days, predictions))


        # 1. Plot Historical Data with SMAs using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add historical closing prices
        ax.plot(data.index, data['Close'], label='Historical Prices', color='blue')
        ax.plot(data.index, data['SMA_50'], label='50-Day SMA', color='orange', linestyle='--')
        ax.plot(data.index, data['SMA_200'], label='200-Day SMA', color='green', linestyle='--')
        ax.set_title("Yes Bank Historical Stock Price & SMAs")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        fig.autofmt_xdate()  # Rotate date labels for better readability

        # Save the historical plot as an image
        historical_image_path = os.path.join('static', 'images', 'historical_plot.png')
        os.makedirs(os.path.dirname(historical_image_path), exist_ok=True)
        fig.savefig(historical_image_path)
        plt.close(fig)
        # Predicted Price
        # 2. Plot Predicted Prices using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(next_x_days, predictions, label='Predicted Prices', color='red', marker='o', linestyle='-', markersize=6)

        ax.set_title("Yes Bank Predicted Stock Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        fig.autofmt_xdate()  # Rotate date labels for better readability

        # Save the predicted plot as an image
        predicted_image_path = os.path.join('static', 'images', 'predicted_plot.png')
        fig.savefig(predicted_image_path)
        plt.close(fig)
    

        return render_template('result.html', 
                               predicted_data=predicted_data, 
                               current_price=current_price,
                               historical_image_path=historical_image_path,
                               predicted_image_path=predicted_image_path)
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
    # 192.168.1.11
