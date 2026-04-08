import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import math
import scipy.optimize as sco
import sys
from optimize import optimize, compare
from metrics import compute_metrics
from recs import generate_recs
from tab3 import features, model, risk_model
import plotly.express as px
import plotly.graph_objects as go


def main():
    # Make page content full width, must be first streamlit call in program
    st.set_page_config(layout="wide")
     
    # Adjust padding around page content to make space above header smaller
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>            
    """, unsafe_allow_html=True)
    
    # Make sizing of font in metrics smaller
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
    """, unsafe_allow_html=True)
    style_metric_cards(background_color="#fdfdf8", border_color="##cb785c", border_left_color="#cb785c")
    
    # Website title and site organization
    st.title("Investment Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Portfolio Optimization", "Portfolio Classification", "Future Allocation"])
    
    # Content in Overview tab
    with tab1:
        #Ask user to upload a csv file of their stocks
        st.subheader("File Upload", divider="gray")
        #upload = st.file_uploader("""Upload a CSV file containing a stock portfolio using the following format for each item: 
        #        ticker,shares,total_cost_basis,acquisition_date""")
        test_val = True
        upload = "holdings.csv"

        #If upload is successful, read the csv into a dataframe named portfolio
        if test_val:
            portfolio = create_portfolio(upload)
            
            price_data_df = create_price_data(portfolio)
            
            #Calculate daily returns
            returns_df = price_data_df.pct_change()
            
            #Compute portfolio and performance metrics
            weights, initial_val, portfolio_returns = compute_metrics(price_data_df, portfolio, returns_df)

            #Display portfolio
            st.header("Your Portfolio", divider="gray")
            port_rename = portfolio.rename(columns={
                'ticker': 'Ticker',
                'shares': 'Shares',
                'total_cost_basis': "Total Cost Basis",
                'acquisition_date': "Acquisition Date",
                "avg_cost": "Average Cost",
                "current_price": "Current Price",
                "current_value": "Current Value",
                "gain_loss": "Total Change",
                "weight": "Weight"
            })
            st.dataframe(port_rename.round(2), hide_index=True)
            
            #Display total gain/loss for each stock and cumulatively
            get_gain_loss(portfolio, returns_df, portfolio_returns)
            
            #Display 2 columns for price data and returns (raw data)
            col1, col2 = st.columns(2)
            col1.header("Stock Data", divider="gray")
            col1.write(price_data_df)
            col2.header("Returns", divider="grey")
            col2.write(returns_df)
            
    # Tab 2 Content
            with tab2:
                col8, col9 = st.columns([1, 2])
                #Calculate optimized weights for portfolio using Sharpe ratio
                col8.subheader("Define Max Allocation Per Stock (%)", help="Move the slider below to see what the optimal portfolio would be given a maximum amount a stock can hold. Adding a max allocation for each stock helps with diversification, which can minimize risk and increase return across your portoflio.")
                max_bound = col8.slider(
                    "Drag slider to adjust max allocation",
                    min_value=10,  # 10%
                    max_value=100, # 100%
                    value=50,      # Default 50%
                    step=5         # Values increment by 5% each time
                )
                
                # Display current metrics next to max allocation slider
                col9.subheader("Current Statistics", help="Statistics for your current portfolio")
                current_metrics = compare(returns_df, weights, initial_val)
                curr_final, curr_sharpe = disp_current(current_metrics, col9)
                
                col10, col11 = st.columns(2)
                max_bound_frac = max_bound / 100
                frontier_data = optimize(returns_df, portfolio, initial_val, weights, max_alloc=max_bound_frac)
                frontier_df = pd.DataFrame(frontier_data)
                col10.header("Efficient Frontier", help="Calculated set of optimal portfolios with the lowest volatility given a baseline level of expected return.", divider="gray")
                col10.line_chart(frontier_df.set_index("volatility")["total_return"])
                
                weights_df = pd.DataFrame(
                    frontier_df["weights"].tolist(),
                    columns=portfolio["ticker"]
                )
                
                optimal_weights = frontier_df.iloc[-1]["weights"]
                optimal_weights = pd.Series(optimal_weights, index=portfolio["ticker"], name="Optimal Weights")
                current_weights = pd.Series(weights, index=portfolio["ticker"], name="Current Weights")
                weights_compare_df = pd.concat([current_weights, optimal_weights], axis=1)
                col11.header("Current vs. Optimal Weights", help="Current vs. optimal weights for portfolio, assuming the optimal portfolio has the highest return value.", divider="gray")
                col11.bar_chart(weights_compare_df)

                display_df = get_eff_frontier_data(frontier_df, weights_df)
                st.header("Efficient Frontier Data", help="Statistics and values of portfolios on the efficient frontier, calculated using the Markowitz theory (MPT). The most optimal portfolios were ones with minimal volatility at different target return rates (acceptable baselines for an expected rate of return).", divider="gray")
                st.dataframe(display_df.round(3), hide_index=True)
                
                st.header("Portfolio Recommendations", help="Recommendations for shares of stocks to buy/sell based on portfolios from the Efficient Frontier with higher Sharpe ratios or larger returns.", divider="gray")
                col12, col13 = st.columns(2)
                recs_sharpe, recs_ret, sharpe_row, ret_row = generate_recs(portfolio, returns_df, frontier_df, current_weights, initial_val)
                
                col12.subheader("For a higher Sharpe ratio:", help="These recommendations could be helpful in increasing your portfolio's Sharpe ratio by increasing the ratio of returns to volatility. For a portfolio, a Sharpe ratio > 1 is good, and portfolio diversification can help increase a Sharpe ratio.")
                with col12:
                    if sharpe_row["sharpe"] < curr_sharpe:
                        st.write("No recommendations available due to current portfolio's Sharpe ratio being greater than portfolio options from the Efficient Fronter. To explore more options, move the slider for max allocation.")
                    else:
                        recs_as_metric(recs_sharpe, portfolio["ticker"])
                
                col13.subheader("For a higher return:", help="These recommendations could be helpful in increasing your portfolio's current value, but they don't account for added volatility. Since stocks with higher returns typically also come with increased volatility, focusing only on risk-unadjusted returns like these could potentially decrease your Sharpe ratio.")
                with col13:
                    if ret_row["final_value"] < curr_final:
                        st.write("No recommendations available due to current portfolio's value being greater than portfolio options from the Efficient Frontier. To explore more options, move the slider for max allocation.")
                    else:
                        recs_as_metric(recs_ret, portfolio["ticker"])
         
    # Tab 3 Content    
            with tab3:
                # ADD SOME STREAMLIT LOADING 
                st.header("Portfolio Features", divider="gray", help="Features of portfolio created by combining features of its stocks and funds by their weight.")
                ticker_features = features.get_ticker_info(portfolio['ticker'])
 
                beta, volatility, momentum, sector_fig, breakdown_fig = classify_portfolio(portfolio['ticker'], weights, ticker_features)
                monthly_vol, month_change = create_risk_model(portfolio['ticker'], weights)
                
                format_tab3(beta, volatility, momentum, sector_fig, breakdown_fig, monthly_vol, month_change)
                
                
            
def create_portfolio(upload_file):
    portfolio = pd.read_csv(upload_file)
    portfolio["avg_cost"] = portfolio["total_cost_basis"] / portfolio["shares"]
    portfolio["acquisition_date"] = pd.to_datetime(portfolio["acquisition_date"])
    portfolio["acquisition_date"] = portfolio["acquisition_date"].dt.date
    
    return portfolio

@st.cache_data
def create_price_data(portfolio):
    price_data = {}
    for idx, row in portfolio.iterrows():
        ticker = row["ticker"]
        start = row["acquisition_date"].strftime("%Y-%m-%d")
                
        data = yf.download(
                ticker,
                start=start,
                auto_adjust=True
        ) ["Close"]

        if data.empty:
            st.warning(f"No data for {ticker}, skipping.")
        else:
            price_data[ticker] = data
            
        #Store the historical data in a dataframe named price_data_df
    price_data_df = pd.concat(price_data, axis=1)
    price_data_df.columns = price_data_df.columns.get_level_values(0)
    price_data_df.ffill(inplace=True)
    
    return price_data_df

def get_gain_loss(portfolio, returns_df, portfolio_returns):
    tickers = portfolio["ticker"].tolist()
    cols = st.columns(len(tickers) + 1)
    
    for i, ticker in enumerate(tickers):
        row = portfolio[portfolio["ticker"] == ticker].iloc[0]
        gain_loss = row["gain_loss"]
        percent_gain = gain_loss / row["total_cost_basis"] * 100
        sparkline = returns_df[ticker].dropna()
        
        cols[i].metric(
            label=ticker,
            value=f"${gain_loss:,.2f}",
            delta=f"{percent_gain:.2f}%",
            chart_data=sparkline,
            border=True
        )
        
    total_gain_loss = portfolio["gain_loss"].sum()
    total_cost = portfolio["total_cost_basis"].sum()
    total_percent = total_gain_loss / total_cost * 100

    cols[-1].metric(
        label="Cumulative",
        value=f"${total_gain_loss:,.2f}",
        delta=f"{total_percent:.2f}%",
        chart_data=portfolio_returns,
        border=True
    )
    
def disp_current(metrics, col):
    initial_val = round(metrics["initial_value"], 2)
    curr_final = round(metrics["current_final"], 2)
    curr_ret = round(metrics["current_return"], 2)
    curr_vol = round((metrics["current_vol"] * 100), 2)
    curr_sharpe = round(metrics["current_sharpe"], 2)
    
    col1, col2, col3, col4 = col.columns(4)
    col1.metric("Initial Value", f"${initial_val}", border=True, help="Initial value of portfolio")
    col2.metric("Value", f"${curr_final}", f"{curr_ret}%", border=True, help="Value and return of current portfolio")
    col3.metric("Volatility", f"{curr_vol}%", border=True, help="Historical volatility of portfolio ")
    col4.metric("Sharpe", f"{curr_sharpe}", border=True, help="Sharpe ratio of current portfolio")
    
    return curr_final, curr_sharpe

def get_eff_frontier_data(frontier_df, weights_df):   
    display_df = pd.concat([
        frontier_df.drop(columns=["weights"]),
        weights_df
    ], axis=1)
                
    display_df["target_return"] = (display_df["target_return"] * 100).round(2).astype(str) + "%"
    display_df["total_return"] = (display_df["total_return"] * 100).round(2).astype(str) + "%"
    display_df["volatility"] = (display_df["volatility"] * 100).round(2).astype(str) + "%"
    display_df["return_diff"] = (display_df["return_diff"] * 100).round(2).astype(str) + "%"
    display_df["final_value"] = display_df["final_value"].apply(lambda x: f"${x:,.2f}")
    display_df["sharpe"] = display_df["sharpe"].round(2)
                
    display_df = display_df.rename(columns={
        "target_return": "Target Return",
        "volatility": "Volatility",
        "final_value": "Value",
        "total_return": "Total Return", 
        "return_diff": "Return Difference",
        "sharpe": "Sharpe"
    })
    
    return display_df

def recs_as_metric(recs_list, tickers):
    len_tickers = len(tickers)
    num_rows = math.ceil(len_tickers / 2)
    t_idx = 0
    
    for row in range(num_rows):
        cols = st.columns(2)
        for col in cols:
            if t_idx < len_tickers:
                ticker = tickers[t_idx]
                action, reason = recs_list[t_idx]
                
                expander_label = f"{action} of {ticker}"
            
                with col:
                    with st.expander(label=expander_label):
                        st.write(reason)

                t_idx += 1
            else:
                col.empty()

@st.cache_data
def classify_portfolio(tickers, weights, features_list):
    beta, volatility, momentum, sectors, breakdown = model.get_portfolio_classification(tickers, weights, features_list)
    
    # Label sector names from sector vector passed in
    sector_names = ["Technology", "Financial Services", "Consumer Cyclical", "Communication Services", "Healthcare", "Industrials", 
               "Energy", "Consumer Defensive", "Basic Materials", "Utilities", "Real Estate"]
    sectors_dict = dict(zip(sector_names, sectors.tolist()))
    sectors_df = pd.DataFrame(list(sectors_dict.items()), columns=["Sector", "Percent"])
    sectors_df["Percent"] = (sectors_df["Percent"] * 100).round(3)
    sector_fig = px.pie(sectors_df, values="Percent", names="Sector", 
                    color_discrete_sequence=["#40826D", "#93C572", "#478778",
                                             "#50C878", "#AFE1AF", "#AAFF00",
                                             "#4F7942", "#00A36C", "#98FB98",
                                             "#B4C424", "#009E60"])
    sector_fig.update_traces(textinfo="label")
    sector_fig.update_traces(hovertemplate="%{label}: %{value:.3f}%")
    sector_fig.update_layout(
        legend=dict(
            x=1,
            xanchor='left',
            y=1,
            yanchor='top'
        )
    )
    
    
    # Label breakdown indices with their investment style
    label_map = {
        0: "Blend", 
        1: "Growth", 
        2: "Value"
    }
    labeled_breakdown = {
        label_map[k]: float(np.squeeze(v)) for k, v in breakdown.items()
    }
    breakdown_df = pd.DataFrame(list(labeled_breakdown.items()), columns=["Style", "Percent"])
    breakdown_df["Percent"] = (breakdown_df["Percent"] * 100).round(3)
    breakdown_fig = go.Figure()
    breakdown_fig = px.pie(breakdown_df, values="Percent", names="Style", 
                           color_discrete_sequence=["#6F8FAF", "#89CFF0", "#0F52BA"])
    breakdown_fig.update_traces(textinfo="label")
    breakdown_fig.update_traces(hovertemplate="%{label}: %{value:.3f}%")
    breakdown_fig.update_layout(
        legend=dict(
            x=1,
            xanchor='left',
            y=1,
            yanchor='top'
        )
    )
    
    return beta, volatility, momentum, sector_fig, breakdown_fig

def format_tab3(beta, vol, momentum, sector_fig, breakdown_fig, monthly_vol, month_change):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta", f"{beta:.3f}", border=True, help="A beta measures how volatile an investment is compared to the entire market. The portfolio's beta was computed through the summation of the beta values of each individual investment multiplied by its weight in the portfolio.")
    col2.metric("Annual Volatility", f"{(vol*100):.2f}%", border=True, help="Volatility was computed through the summation of each individual investment's volatility multiplied by its weight.")
    col3.metric("Momentum", f"{momentum:.3f}", border=True, help="Momentum measures the recent performance of an investment, and this value was computed through the summation of the momentum from each individual investment multiplied by its weight.")
    col4.metric("Next Month's Expected Volatility", f"{(monthly_vol * 100):.2f}%", f"{month_change:.2f}%", border=True) 
    
    col4, col5 = st.columns([1.75, 1], border=True)
    with col4:
        st.subheader("Portfolio Sector Breakdown", divider="gray", help="Sector breakdown of entire portfolio.")
        st.plotly_chart(sector_fig)
    
    with col5:
        st.subheader("Portfolio Classification", divider="gray", help="Investments can be classified by their growth potential, valuation, or a mix of both. The below classification was created using a Random Forest algorithm, where each individual investment was classified by a model using its features.")
        st.plotly_chart(breakdown_fig)
        st.caption("""There are three categories for classifying the styles towards which investment instruments are oriented: growth, value, and blended. 
                   The growth style is known for its capital gains potential, where their value is expected to grow sales and earnings at a faster rate than the market average; 
                   however, this style's high returns also makes it risky. The value style is known for its stability and dividend income, offering long-term growth opportunities.
                   The blended style combines both growth and value investments, which offers diversification benefits due to its mix of capital gains and stability.
                   """)
             
         

def create_risk_model(tickers, weights):
    monthly_vol, month_change = risk_model.get_risk_model(tickers, weights)
    
    return monthly_vol, month_change

    
if __name__=="__main__":
    main()
    