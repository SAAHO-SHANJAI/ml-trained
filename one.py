import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from PyEMD import CEEMDAN
import os
from sklearn.metrics import r2_score

# Function to remove outliers using IQR
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Function to build optimized model
def build_model(lookback):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(lookback, 1), padding='same'),
        MaxPooling1D(2),
        LSTM(100, return_sequences=True, kernel_regularizer='l2'),
        Dropout(0.5),
        LSTM(50),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss=Huber(delta=1.0))
    return model

# Function to create sequences
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# Load and preprocess data
file_path = 'data/Opeloop_HFc_TrTj(1).xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Excel file not found at: {file_path}")
df = pd.read_excel(file_path)
data = df.iloc[:, 0].values.reshape(-1, 1)
data = remove_outliers(data).reshape(-1, 1)  # Remove outliers

# Apply CEEMDAN
ceemdan = CEEMDAN()
imfs = ceemdan(data.flatten())
n_imfs = imfs.shape[0]

# Filter relevant IMFs (exclude first and last IMF)
filtered_imfs = imfs[1:-1] if n_imfs > 2 else imfs

predictions_imf = []
lookback = 100  # Use larger lookback
scaler = MinMaxScaler()

for i in range(len(filtered_imfs)):
    current_data = filtered_imfs[i].reshape(-1, 1)
    normalized_data = scaler.fit_transform(current_data)
    X, y = create_sequences(normalized_data, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * 0.85)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = build_model(lookback)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    test_predict = model.predict(X_test, verbose=0)
    test_predict = scaler.inverse_transform(test_predict)
    predictions_imf.append(test_predict)

# Combine predictions
final_predictions = np.sum(predictions_imf, axis=0).flatten()
y_actual = data[lookback:].flatten()

# Ensure same length
y_actual = y_actual[:len(final_predictions)]

# Calculate RMSE and R²
rmse = np.sqrt(np.mean((final_predictions - y_actual) ** 2))
r2 = r2_score(y_actual, final_predictions)
print(f"Final RMSE: {rmse:.3f}")
print(f"Final R²: {r2:.3f}")

# Plot results
plt.figure(figsize=(15, 8))
time_points = np.arange(len(y_actual))
plt.plot(time_points, y_actual, 'b-', label='Actual', linewidth=1.5)
plt.plot(time_points, final_predictions, 'r--', label='Predicted', linewidth=1.5)
plt.axvline(x=int(len(y_actual) * 0.8), color='g', linestyle='--', label='Train-Test Split')
plt.title(f'Actual vs Predicted Values\nRMSE={rmse:.3f}, R²={r2:.3f}', fontsize=12)
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

# Plot error distribution
plt.figure(figsize=(10, 6))
error = y_actual - final_predictions
plt.hist(error, bins=50, color='blue', alpha=0.7)
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid()
plt.show()
