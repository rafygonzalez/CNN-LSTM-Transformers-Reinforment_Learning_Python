import math

from sklearn.preprocessing import MinMaxScaler
import quandl
import matplotlib.pyplot as plt
# Import required libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



# Get Apple stock data from Quandl
data = quandl.get("WIKI/AAPL",
                  trim_start = "2010-01-01", 
                  trim_end = "2021-01-01")

# Extract the closing prices from the data
closing_prices = data["Adj. Close"]


values = closing_prices.values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))

train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.Sequential()
model.add(layers.LSTM(600, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(600, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()
print(x_test)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=3, epochs=3)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) **2)

train = data[:training_data_len]
validation = data[training_data_len:]

validation['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(validation[['Adj. Close', 'Predictions']])
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()

# Plot the closing prices on a graph
plt.plot(closing_prices)
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Apple Stock Price")
plt.show()

