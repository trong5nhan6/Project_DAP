import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model(stock_name, csv_path, model_save_dir='model/checkpoint', window_size=30):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    close_prices = df['Close'].values

    X, y = [], []
    for i in range(len(close_prices) - window_size):
        X.append(close_prices[i:i + window_size])
        y.append(close_prices[i + window_size])

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{stock_name.lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

if __name__ == "__main__":
    for i in ["gg", "netflix", "apple", "nvidia", "tesla"]:
        train_and_save_model(i, f"data/{i}.csv")
