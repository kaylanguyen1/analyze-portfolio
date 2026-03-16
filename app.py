import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

def main():
    st.set_page_config(layout="wide")
    st.title("Investment Dashboard")
    tab1, tab2, tab3 = st.tabs(["Overview", "Tab 2", "Tab 3"])
    
    with tab1:
        #Ask user to upload a csv file of their stocks
        st.subheader("File Upload", divider="gray")
        upload = st.file_uploader("Upload a portfolio CSV:")

        #If upload is successful, read the csv into a dataframe named portfolio
        if upload:
            portfolio = pd.read_csv(upload)
            portfolio["avg_cost"] = portfolio["total_cost_basis"] / portfolio["shares"]
            portfolio["acquisition_date"] = pd.to_datetime(portfolio["acquisition_date"])
            portfolio["acquisition_date"] = portfolio["acquisition_date"].dt.date
            
            #Access yfinance to collect historical data for all stocks in portfolio starting at their bought date
            price_data = {}
            for idx, row in portfolio.iterrows():
                ticker = row["ticker"]
                start = row["acquisition_date"].strftime("%Y-%m-%d")
                
                data = yf.download(
                    ticker,
                    start=start,
                    auto_adjust=True
                ) ["Close"]

                price_data[ticker] = data
            
            #Store the historical data in a dataframe named price_data_df
            price_data_df = pd.concat(price_data, axis=1)
            price_data_df.columns = list(price_data.keys())
            price_data_df.ffill(inplace=True)
            
            #Calculate daily returns
            returns = price_data_df.pct_change()
            
            #Compute portfolio and performance metrics
            compute_metrics(price_data_df, portfolio, returns)
            
            #Display previously created dataframes
            st.header("Your Portfolio", divider="gray")
            st.write(portfolio.round(2))
            
            col1, col2 = st.columns(2)
            col1.header("Stock Data", divider="gray")
            col1.write(price_data_df)
            col2.header("Returns", divider="grey")
            col2.write(returns)
    
def compute_metrics(price_data_df, portfolio, returns):
    #Standard metrics
    latest_prices = price_data_df.iloc[-1]
    portfolio["current_price"] = portfolio["ticker"].map(latest_prices)
    portfolio["current_value"] = portfolio["shares"] * portfolio["current_price"] 
    total_value = portfolio["current_value"].sum()
    portfolio["gain_loss"] = portfolio["current_value"] - portfolio["total_cost_basis"]
    portfolio["weight"] = portfolio["current_value"] / total_value
    weights = portfolio.set_index("ticker")["weight"]
    portfolio_returns = (returns.fillna(0) * weights).sum(axis=1)
    
    #Performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / volatility
    initial_val = total_value
    cumulative_val = (1 + portfolio_returns).cumprod() * initial_val
    cum_val_df = cumulative_val.to_frame(name="Cumulative Value")
    
    #Stock plot of all stocks and their valuation through time
    stock_values_df = price_data_df * portfolio.set_index("ticker")["shares"]
    stock_values_df = pd.concat([stock_values_df, cum_val_df], axis=1)
    st.header("Stock Chart", divider="gray")
    st.line_chart(stock_values_df, x_label="Date", y_label="Value ($)")
    
if __name__=="__main__":
    main()
    