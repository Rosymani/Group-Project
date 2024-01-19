#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:19:38 2024

@author: surendrakumarreddypolaka
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load your time series data from a CSV file
# Make sure your CSV file has columns: 'timestamp', 'a_left_backPose', 'b_left_backPose', 'c_left_backPose', 'File Name'
data = pd.read_csv('/Users/surendrakumarreddypolaka/Desktop/Group-Project-main/my_data_left_backPose_s01.csv')

# Extract relevant columns for time series (e.g., 'a_left_backPose')
time_series_columns = ['timestamp', 'a_left_backPose', 'b_left_backPose', 'c_left_backPose']
time_series_data = data[time_series_columns]

# Function to check stationarity using Augmented Dickey-Fuller test
def check_stationarity(column_data):
    result = adfuller(column_data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')

# Normalize the data using StandardScaler for each column
scalers = {}
for column in time_series_columns[1:]:
    scaler = StandardScaler()
    time_series_data[column] = scaler.fit_transform(time_series_data[column].values.reshape(-1, 1))
    scalers[column] = scaler

# Check stationarity for each column
for column in time_series_columns[1:]:
    print(f"Checking stationarity for column: {column}")
    check_stationarity(time_series_data[column].values)

# Split the data into training and testing sets
train_size = int(len(time_series_data) * 0.8)
train_data, test_data = time_series_data[:train_size], time_series_data[train_size:]

# Create sequences and labels for training
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length, 1:]  # Exclude 'timestamp' column
        label = data.iloc[i + seq_length, 1:]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 10  # adjust as needed
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(seq_length, len(time_series_columns)-1)))
model.add(Dense(units=len(time_series_columns)-1))  # Adjust units based on the number of columns
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, mae, mse = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Mean Absolute Error: {mae}, Mean Squared Error: {mse}")

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

