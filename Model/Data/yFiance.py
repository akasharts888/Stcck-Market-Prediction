

import yfinance as yf
ticker = 'YESBANK.NS'  # For NSE
testData = yf.download(ticker, start='2020-01-01', end='2024-11-21')
print(testData.index[-1])
print(testData.columns)
print(testData.tail())