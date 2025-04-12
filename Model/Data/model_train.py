import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
ticker = 'YESBANK.NS'  # For NSE
testData = yf.download(ticker, start='2020-01-01', end='2024-11-21')
# # Save the entire model (architecture, weights, and optimizer state)
# data = pd.read_csv('Model\Data\Bank_Stock.csv')
testData = testData.iloc[:, 4:5]
# print(testData.head())
sc = MinMaxScaler(feature_range=(0, 1))
testData = sc.fit_transform(testData)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
# print(len(testData))
x_train = []
y_train = []
for i in range(60, len(testData)):
    x_train.append(testData[i-60:i, 0])
    y_train.append(testData[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, shuffle=False
)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
model = tf.keras.models.load_model("Model/lstm_stock_prediction_model.h5")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50,batch_size = 32,callbacks=[early_stopping])
test_loss, test_mae, test_mse = model.evaluate(x_val, y_val)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test MSE: {test_mse}")

# model.save('Model/lstm_stock_prediction_model.h5')