import numpy as np
import pandas as pd
import streamlit as st

def generate_recs(portfolio, returns_df, frontier_df, weights, initial_val):
    sharpe_row = frontier_df.loc[frontier_df["sharpe"].idxmax()]
    ret_row = frontier_df.loc[frontier_df["total_return"].idxmax()]
    
    weights_sharpe = pd.Series(sharpe_row["weights"], index=portfolio["ticker"])
    weights_ret = pd.Series(ret_row["weights"], index=portfolio["ticker"])
    
    recs_sharpe = []
    recs_ret = []
    
    for ticker in portfolio["ticker"]:
        diff_sharpe = (weights_sharpe[ticker] - weights[ticker]) * initial_val
        diff_ret = (weights_ret[ticker] - weights[ticker]) * initial_val
        
        stock_price = portfolio.loc[portfolio["ticker"] == ticker, "current_price"].values[0]
        diff_sharpe_shares = diff_sharpe / stock_price
        diff_ret_shares = diff_ret / stock_price
        
        hist_ret = returns_df[ticker].mean() * 252
        hist_vol = returns_df[ticker].std() * np.sqrt(252)
        risk_free_rate = 0.03
        indiv_sharpe = (hist_ret - risk_free_rate) / hist_vol
        
        if abs(diff_sharpe_shares) > 0.01:
            if diff_sharpe_shares > 0:
                action = f"Buy {abs(diff_sharpe_shares):.2f} shares"
                reason = f"This stock has a Sharpe ratio of {indiv_sharpe:.2f}, a historical return rate of {hist_ret:.2%}, and a volatility of {hist_vol:.2%}. Since the difference between this stock's current and optimal weight is positive, increasing this stock could improve the overall portfolio Sharpe ratio due to diversification benefits."
            else:
                action = f"Sell {abs(diff_sharpe_shares):.2f} shares"
                reason = f"This stock has a Sharpe ratio of {indiv_sharpe:.2f}, a historical return rate of {hist_ret:.2%}, and a volatility of {hist_vol:.2%}. Since the difference between this stock's current and optimal weight is negative, decreasing this stock could improve the overall portfolio Sharpe ratio."
            recs_sharpe.append([action, reason])
        
        if abs(diff_ret_shares) > 0.01:
            if diff_ret_shares > 0:
                action = f"Buy {abs(diff_ret_shares):.2f} shares"
                reason = f"Has historical return rate of {hist_ret:.2%} which could improve portfolio performance, but could also increase portfolio's volatility (volatility = {hist_vol:.2%})."
            else:
                action = f"Sell {abs(diff_ret_shares):.2f} shares"
                reason = f"This stock has a historical return rate of {hist_ret:.2%} and a volatility of {hist_vol:.2%}. Although its return rate might not be as explosive, some stocks with smaller returns can be used to help with long-term growth."

            recs_ret.append([action, reason])
    return recs_sharpe, recs_ret, sharpe_row, ret_row
    
    