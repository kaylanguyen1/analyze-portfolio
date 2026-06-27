import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st
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

def get_asset_type(tickers, ticker_info):
    features = []
    for ticker in tickers:
        idx = tickers.index(ticker)
        var = yf.Ticker(ticker)
        info = var.info
        asset_type = info.get("quoteType", "N/A")

        if asset_type == "EQUITY":
            features.append(get_stock_features(info, ticker, ticker_info[idx]))
        else:
            features.append(get_fund_features(info, ticker, ticker_info[idx]))
    return features

def get_stock_features(info, ticker, ticker_info):
    stock_features = {}
    stock_features["ticker"] = ticker
    stock_features["momentum"] = ticker_info["momentum"]
    stock_features["volatility"] = ticker_info["volatility"]
    stock_features["beta"] = ticker_info["beta"]
    stock_features["earnings_growth"] = info.get("earningsGrowth", "N/A")
    stock_features["revenue_growth"] = info.get("revenueGrowth", "N/A")
    pe = info.get("forwardPE") if info.get("forwardPE") else info.get("trailingPE")
    stock_features["pe_ratio"] = pe
    
    return stock_features
    
def get_fund_features(info, ticker, ticker_info):
    fund_features = {}
    fund_features["ticker"] = ticker
    fund_features["momentum"] = ticker_info["momentum"]
    fund_features["volatility"] = ticker_info["volatility"]
    fund_features["beta"] = ticker_info["beta"]
    fund_features["expense_ratio"] = info.get("annualReportExpenseRatio", "N/A")
    if fund_features["expense_ratio"] == "N/A":
        expense_ratio = info.get("netExpenseRatio", "N/A")
        if expense_ratio != "N/A":
            fund_features["expense_ratio"] = expense_ratio / 100
    
    var = yf.Ticker(ticker)
    top_holdings = var.funds_data.top_holdings
    if top_holdings is not None:
        top_holdings = top_holdings['Holding Percent'].sum()
    fund_features["diversification_score"] = float(1 - top_holdings)
    
    return fund_features

def normalize_features(df, feature_cols):
    X = df[feature_cols].copy()
    
    for col in feature_cols:
        if col == "is_stock":
            continue
        else:
            mean = X[col].mean()
            std = X[col].std() + 1e-6
            
            X[col] = (X[col] - mean) / std
            X[col] = X[col].clip(-3, 3)
    return X

def build_matrix(feature_list):
    df = pd.DataFrame(feature_list).set_index("ticker")
    all_features = ["momentum", "volatility", "beta", "earnings_growth", "revenue_growth", "pe_ratio", "expense_ratio", "diversification_score", "is_stock"]
    for col in all_features:
        if col not in df.columns:
            df[col] = np.nan

    df["is_stock"] = df["pe_ratio"].notna().astype(int)
    df["earnings_growth"] = np.tanh(df["earnings_growth"])
    df["expense_ratio"] = df["expense_ratio"].replace("N/A", np.nan)
    df["expense_ratio"] = df["expense_ratio"].fillna(0)
    df["diversification_score"] = df["diversification_score"].fillna(0)
    df["earnings_growth"] = df["earnings_growth"].fillna(df["earnings_growth"].mean())
    df["revenue_growth"] = df["revenue_growth"].fillna(df["revenue_growth"].mean())
    df["pe_ratio"] = df["pe_ratio"].fillna(df["pe_ratio"].median())
    
    df = normalize_features(df, all_features)
    
    return df, all_features

# add cov_mtx and weights to arguments when running
def get_score(df, cols, style, strength):
    scaled_weights = weight_dicts(style, strength)
    print("scaled_weights", scaled_weights)
    weight_vector = np.array([scaled_weights.get(col, 0) for col in cols])
    base_score = df[cols] @ weight_vector
    print("base_score", base_score)
    
    df['value_score'] = 0.5 * (1 / df['pe_ratio'].clip(lower=1)) + 0.3 * df['earnings_growth'] + 0.2 * df['revenue_growth']
    df['value_score'] = df['value_score'].rank(pct=True)
    df['value_score'] = (df['value_score'] - 0.5) * 2
    df['growth_score'] = 0.4 * df['earnings_growth'] + 0.3 * df['revenue_growth'] + 0.3 * df['momentum']
    df['growth_score'] = np.tanh(df['growth_score'])
    df['growth_score'] = (df['growth_score'] - df['growth_score'].mean()) / (df['growth_score'].std() + 1e-6)
    print(f"growth_score: {df['growth_score']}, value_score: {df['value_score']}")
    value_weight = {"growth": 0.05, "value": 0.25, "blend": 0.12}[style]
    growth_weight = {"growth": 0.25, "value": 0.05, "blend": 0.12}[style]
    final_score = base_score + value_weight * df['value_score'] + growth_weight * df['growth_score']
    final_score = final_score * 1.2
    
    #if cov_mtx is not None and weights is not None:
    #   marginal_risk = cov_mtx @ weights
    #   marginal_risk = (marginal_risk - marginal_risk.mean()) / (marginal_risk.std() + 1e-6)
    #   final_score -= 0.3 * marginal_risk
    
    return final_score
    
def weight_dicts(style, strength):
    base_weights = {
        "momentum": 0.2, "volatility": -0.2, "beta": -0.1, "earnings_growth": 0.3, "revenue_growth": 0.2,
        "pe_ratio": -0.2, "expense_ratio": -0.2, "diversification_score": 0.2
    }
    
    growth_weights = {
        "momentum": 0.3, "volatility": -0.1, "beta": 0.15, "earnings_growth": 0.35, "revenue_growth": 0.25, 
        "pe_ratio": -0.1, "expense_ratio": -0.02, "diversification_score": 0.1
        }
    
    value_weights = {
        "momentum": 0.02, "volatility": -0.3, "beta": -0.2, "earnings_growth": 0.1, "revenue_growth": 0.08, 
        "pe_ratio": -0.25, "expense_ratio": -0.12, "diversification_score": 0.25
    }
    
    blend_weights = {
        "momentum": 0.18, "volatility": -0.2, "beta": -0.05, "earnings_growth": 0.22, "revenue_growth": 0.15, 
        "pe_ratio": -0.18, "expense_ratio": -0.08, "diversification_score": 0.15
    }
    
    if style == "growth":
        style_weights = growth_weights
    elif style == "value":
        style_weights = value_weights
    else:
        style_weights = blend_weights
    
    return scale_weights(base_weights, style_weights, strength)

def scale_weights(base_weights, style_weights, strength):
    final_weights = {}
    
    for key in base_weights.keys():
        base = base_weights[key]
        target = style_weights.get(key, 0)
        
        final_weights[key] = base + strength * (target - base)
    
    return final_weights

def compute_strength(beta, vol, growth, style, contrib):
    current = 0.5 * growth + 0.3 * vol + 0.2 * beta
    target = {"growth": 1.2, "blend": 1.0, "value": 0.8}
    gap = target[style] - current
    strength = 1 + 0.5 * gap
    strength = np.clip(strength, 0.7, 1.3)
    
    min_contrib = 500
    max_contrib = 10000
    norm = (contrib - min_contrib) / (max_contrib - min_contrib)
    factor = 0.85 + 0.3 * norm
    
    final_strength = strength * factor
    return np.clip(final_strength, 0.7, 1.4)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
    
def main():
    # import (contrib_amt, style, ticker_features, weights, beta, vol, growth, cov_mtx)
    # import ticker_info from app.py and integrate with tickers
    contributions = [500, 1000, 5000, 10000]
    styles = ["growth", "value", "blend"]
    tickers = ['AAPL', 'FELG', 'FNILX', 'FSPGX', 'FXAIX', 'LMT']
    #weights = [0.1587, 0.0291, 0.0757, 0.0327, 0.0654, 0.6386]
    
    momentum = 0.080
    beta = 0.543
    beta = beta - 1
    vol = 0.1133
    vol = vol / 0.3
    growth = 0.6
    
    ticker_info = [
        {'region_US': 1, 'beta': 1.109, 'market_cap_log': 28.967143625860967, 'growth_score': 9.9394176, 'value_score': 0.10060950647555986, 'momentum': 0.013862072844393891, 'volatility': 0.21300694069288315, 'sector_vector': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'turnover': 0.0, 'expense_ratio': 0.0, 'concentration': 1.0},
        {'region_US': 1, 'beta': 1.15, 'momentum': -0.1004255, 'volatility': 0.1725, 'market_cap_log': 25.328436022934504, 'growth_score': 1.0, 'value_score': 0.0, 'sector_vector': [0.53139997, 0.0482, 0.1166, 0.1275, 0.0813, 0.0717, 0.0082, 0.0106, 0.004, 0.00029999999, 0.0], 'expense_ratio': 0.0018000001, 'turnover': 0.6, 'concentration': 0.42948309},
        {'region_US': 1, 'beta': 1.02, 'momentum': -0.0459016, 'volatility': 0.153, 'market_cap_log': 25.328436022934504, 'growth_score': 0.5, 'value_score': 0.5, 'sector_vector': [0.3319, 0.1221, 0.099700004, 0.1073, 0.0994, 0.086899996, 0.0354, 0.0535, 0.0199, 0.024300002, 0.0], 'expense_ratio': 0.0, 'turnover': 0.3, 'concentration': 0.2523083},
        {'region_US': 1, 'beta': 1.16, 'momentum': -0.09768750000000001, 'volatility': 0.174, 'market_cap_log': 25.328436022934504, 'growth_score': 1.0, 'value_score': 0.0, 'sector_vector': [0.4901, 0.0575, 0.13149999, 0.1282, 0.0863, 0.060500003, 0.0042, 0.0299, 0.0038, 0.0032, 0.0], 'expense_ratio': 0.00035, 'turnover': 0.3, 'concentration': 0.411369686},
        {'region_US': 1, 'beta': 1.0, 'momentum': -0.0433704, 'volatility': 0.15, 'market_cap_log': 25.328436022934504, 'growth_score': 0.5, 'value_score': 0.5, 'sector_vector': [0.33080003, 0.122600004, 0.1011, 0.1073, 0.098400004, 0.0866, 0.0348, 0.0543, 0.019299999, 0.0249, 0.0], 'expense_ratio': 0.00014999999, 'turnover': 0.3, 'concentration': 0.25457480000000005},
        {'region_US': 1, 'beta': 0.24, 'market_cap_log': 25.671642776050653, 'growth_score': 5.30546084, 'value_score': 0.18848500472863638, 'momentum': 0.05558289876245537, 'volatility': 0.2705639268854956, 'sector_vector': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'turnover': 0.0, 'expense_ratio': 0.0, 'concentration': 1.0}
    ]
    
    for style in styles:
        for contrib in contributions:
            print("--------------------------------------------------")
            print(f"Style: {style} - Contribution Amount: {contrib}")
            strength = compute_strength(beta, vol, growth, style, contrib)
            features = get_asset_type(tickers, ticker_info)
            features_df, feature_cols = build_matrix(features)
            score = get_score(features_df, feature_cols, style, strength) # add cov_mtx when running
            print(score)
        
            weights = softmax(score)
            weights = np.minimum(weights, 0.3)
            weights /= weights.sum()
            print("weights: ", weights)
    
if __name__=="__main__":
    main()