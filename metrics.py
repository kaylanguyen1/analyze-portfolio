import streamlit as st
import numpy as np
import pandas as pd

def compute_metrics(price_data_df, portfolio, returns_df):
    #Standard metrics
    latest_prices = price_data_df.iloc[-1]
    #Create current price column for portfolio with most recent pricing
    portfolio["current_price"] = portfolio["ticker"].map(latest_prices)
    #Create current value column and calculate total value of portfolio
    portfolio["current_value"] = portfolio["shares"] * portfolio["current_price"] 
    total_value = portfolio["current_value"].sum()
    #Total gain/loss since bought
    portfolio["gain_loss"] = portfolio["current_value"] - portfolio["total_cost_basis"]
    #Calculate weight of portfolio for each stock
    portfolio["weight"] = portfolio["current_value"] / total_value
    weights = portfolio.set_index("ticker")["weight"]
    weights = weights.values
    portfolio_returns = (returns_df.fillna(0) * weights).sum(axis=1)
    
    #Performance of entire portfolio
    initial_val = total_value
    cumulative_val = (1 + portfolio_returns).cumprod() * initial_val
    cum_val_df = cumulative_val.to_frame(name="Cumulative")
    #Stock plot of all stocks and their valuation through time
    stock_values_df = price_data_df * portfolio.set_index("ticker")["shares"]
    stock_values_df = pd.concat([stock_values_df, cum_val_df], axis=1)
    st.header("Stock Chart", divider="gray", help="Portfolio data for individual stocks and their cumulative amount, plotted on their date of acquisition")
    st.line_chart(stock_values_df, x_label="Date", y_label="Value ($)")
    
    return weights, initial_val, portfolio_returns