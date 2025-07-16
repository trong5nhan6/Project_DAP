import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==== CẤU HÌNH ====
TICKERS = ['gg', 'netflix', 'apple', 'nvidia', 'tesla']
WINDOW_SIZE = 30
DATA_DIR = "data"
MODEL_DIR = "model/checkpoint"

def create_dataset(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

for ticker in TICKERS:
    print(f"\n🔁 Đang huấn luyện cho mã: {ticker.upper()}")

    # ==== LOAD DATA ====
    data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file {data_path}, bỏ qua...")
        continue

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    data = df['Close'].values.reshape(-1, 1)

    # ==== SCALING ====
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    model_path = os.path.join(MODEL_DIR, ticker)
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))

    # ==== SLIDING WINDOW ====
    X_all, y_all = create_dataset(scaled, WINDOW_SIZE)

    # ==== LINEAR ====
    X_flat = X_all.reshape(X_all.shape[0], -1)
    linear = LinearRegression()
    linear.fit(X_flat, y_all)
    joblib.dump(linear, os.path.join(model_path, "Linear.pkl"))
    print("Linear xong")

    # ==== RANDOM FOREST ====
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_flat, y_all.ravel())
    joblib.dump(rf, os.path.join(model_path, "RandomForest.pkl"))
    print("Random Forest xong")

    # ==== LSTM ====
    X_seq = X_all.reshape(-1, WINDOW_SIZE, 1)
    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        LSTM(32),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_seq, y_all, epochs=30, batch_size=32,
             callbacks=[EarlyStopping(patience=5)], verbose=1)
    lstm.save(os.path.join(model_path, "LSTM.h5"))
    print("LSTM xong")

print("\n🎉 Huấn luyện xong toàn bộ!")
