import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, Dense, Dropout
from keras import Input
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.layers import Embedding, Bidirectional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TextVectorization

def create_sequences(data, target, window_size=30):
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size]) # 30-day slice of returns
        y.append(target[i+window_size]) # Predict next day volatility
    
    return np.array(X), np.array(y)

def train_model(X, y):
    # Fit scaler for X and y on training data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
        
    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    split = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    
    # Define neural network
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.35),
        LSTM(32),
        Dropout(0.35),
        Dense(1)
    ])
    
    # Optimize with gradient descent 
    model.compile(optimizer="adam", loss="mse")
    
    model.fit(
        X_train, y_train,
        epochs=19,
        batch_size=32, 
        validation_data=(X_test, y_test)
    )
    
    return model, scaler_X, scaler_y


def get_risk_model(tickers, weights):
    predicted_vols = {}
    X_all, y_all = [], []
    ticker_idx = {}
    all_returns = []
    
    start_idx = 0
    for ticker in tickers:
        # Pandas series with index=dates, values=closing prices
        prices = yf.download(ticker, period="5y")["Close"].squeeze()
        # Return log change of price between consecutive days
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Predict 10-day rolling volatility
        volatility = returns.rolling(window=10).std().dropna()
        returns = returns.iloc[9:]
        all_returns.append(returns.rename(ticker))
        
        X, y = create_sequences(returns.values, volatility.values)
        X_all.append(X)
        y_all.append(y)
        
        end_idx = start_idx + len(X)
        ticker_idx[ticker] = (start_idx, end_idx)
        start_idx = end_idx
    
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    model, scaler_X, scaler_y = train_model(X_all, y_all)
    
    X_all_scaled = scaler_X.transform(X_all.reshape(-1, 1)).reshape(X_all.shape)
    
    predicted_vol_all = model.predict(X_all_scaled)
    predicted_vol_all = scaler_y.inverse_transform(predicted_vol_all)
    
    for ticker in tickers:
        start, end = ticker_idx[ticker]
        ticker_preds = predicted_vol_all[start:end]
        predicted_vols[ticker] = float(ticker_preds[-1][0])
        
    print(predicted_vols)
    
    # Turn predicted_vols into vector
    vol_vector = np.array([predicted_vols[t] for t in tickers])
    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    returns_df = pd.concat(all_returns, axis=1).dropna()
    cov_matrix = returns_df.cov()
    for i in range(len(tickers)):
        cov_matrix.iloc[i, i] = vol_vector[i] ** 2
        
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var) # Portfolio's expected daily volatility
    port_vol_annual = port_vol * np.sqrt(252)
    hist_vol = np.sqrt(weights.T @ returns_df.cov() @ weights)
    
    print("annual vol: ", port_vol_annual)
    print("hist vol: ", hist_vol)
    print("Predicted portfolio volatility: ", port_vol)
    
    # Expected volatility of portfolio for next 30 days
    # Daily volatility
    monthly_vol = port_vol * np.sqrt(30) 
    
    portfolio_ret = returns_df @ weights
    last_month_vol = portfolio_ret[-30:].std()
    month_change = (monthly_vol - last_month_vol) / last_month_vol
    print("montly_vol: ", monthly_vol)
    print("last_month_vol: ", last_month_vol)
    print("month_change: ", month_change)
    
    return monthly_vol, month_change   
    
    
