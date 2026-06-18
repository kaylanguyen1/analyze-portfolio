import pandas as pd
import yfinance as yf
import numpy as np
import logging

risk_free = 0.02

def new_ticker_info(input_tickers):
    valid_tickers = verify_tickers(input_tickers)
    ticker_info = {}
    ticker_prices = {}
    
    spy_var = yf.Ticker("SPY")
    spy_metrics, spy_ret, spy_prices = get_metrics(spy_var)
    ticker_info["SPY"] = spy_metrics
    ticker_prices["SPY"] = spy_prices
    
    for ticker in valid_tickers:
        metrics, prices = get_financials(ticker, spy_metrics, spy_ret)
        ticker_info[ticker] = metrics
        ticker_prices[ticker] = prices
        
    return ticker_info, ticker_prices

def verify_tickers(tickers):
    valid_tickers = []
    for t in tickers:
        if is_valid_ticker(t) is True:
            valid_tickers.append(t)
            print(f"{t} is valid")
        else:
            print(f"{t} is invalid")
            
    return valid_tickers
    
# Only return valid tickers from user input and silence terminal output for invalid tickers
def is_valid_ticker(ticker):
    logger = logging.getLogger('yfinance')
    logger.disabled = True
    logger.propagate = False
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="1mo")
    except Exception:
        pass
    
    return not hist.empty

# Get stock/fund data based on quote type, return stats
def get_financials(ticker, spy_metrics, spy_ret):
    var = yf.Ticker(ticker)
    var_info = var.info
    fast_inf = var.fast_info
    quoteType = var_info.get('quoteType', 'N/A')
    if var_info.get('currentPrice'):
        curr_price = var_info.get('currentPrice')
    else:
        curr_price = fast_inf.get('lastPrice')
        
    # Get annual returns, volatility, sharpe, momentum, max drawdowns, and current price
    curr_metrics, log_ret, prices = get_metrics(var)

    # Compare ticker's values against S&P 500
    sharpe_diff, ret_diff, vol_diff, momentum_diff, drawdown_diff = get_spy_diff(curr_metrics, spy_metrics)
    
    if quoteType == "EQUITY":
        stock_score, stock_weight = 0, 0
        roe = var_info.get("returnOnEquity", "N/A")
        peg = var_info.get("pegRatio", "N/A")
        low = var_info.get("fiftyTwoWeekLow", "N/A")
        high = var_info.get("fiftyTwoWeekHigh", "N/A")
        market_score = (0.4 * sharpe_diff) + (0.3 * momentum_diff) + (0.2 * drawdown_diff) + (0.1 * ret_diff)

        sortino_score, roe_score, peg_score, price_score = stock_metric(curr_metrics['annual_ret'], log_ret, roe, peg, low, high, curr_price)
        
        if roe != "N/A":
            stock_score += 0.3 * roe_score
            stock_weight += 0.3
        if peg != "N/A":
            stock_score += 0.25 * peg_score
            stock_weight += 0.25
        if high != "N/A" and low != "N/A":
            stock_score += 0.2 * price_score
            stock_weight += 0.2
        stock_score += 0.25 * sortino_score
        stock_weight += 0.25
        stock_score /= stock_weight
        
        score = (0.7 * market_score) + (0.3 * stock_score)
        signal = get_stock_score(score)
        
    else:
        info_ratio = fund_metric(spy_ret, log_ret)
        score = (0.3 * sharpe_diff) + (0.25 * momentum_diff) + (0.15 * drawdown_diff) + (0.15 * ret_diff) + (0.15 * info_ratio)
        signal = get_fund_score(score)
        
    curr_metrics["score"] = float(score)
    curr_metrics["signal"] = signal
    
    return curr_metrics, prices
        
# Compute annual returns, sharpe ratio, momentum, max drawdown and return + log returns
def get_metrics(ticker):
    hist = ticker.history(period="5y")
    prices = hist["Close"]
    log_ret = np.log(prices).diff().dropna()
    annual_ret = np.exp(log_ret.mean() * 252) - 1
    vol = log_ret.std() * np.sqrt(252)
    
    sharpe = (annual_ret - risk_free) / (vol + 1e-8)
    
    # Momentum can be 12, 6, or 3 month depending on data availability
    if len(prices) > 252:
        momentum = prices.iloc[-1] / prices.iloc[-252] - 1
    elif len(prices) > 126:
        momentum = prices.iloc[-1] / prices.iloc[-252] - 1
    elif len(prices) > 60:
        momentum = prices.iloc[-1] / prices.iloc[-60] - 1
    else:
        momentum = 0
    
    # Raw max drawdown 
    roll_max = prices.cummax()
    drawdown = prices / roll_max - 1
    max_drawdown = abs(drawdown.min())
    
    # Get recommendation from yfinance API to compare
    yf_recs_df = ticker.recommendations_summary
    print(yf_recs_df)
    if yf_recs_df.empty:
        yf_rec = "None"
    else:   
        target_cols = ["strongBuy", "buy", "hold", "sell", "strongSell"]
        yf_rec = yf_recs_df[target_cols].mean().idxmax()
        formatted_recs_dict = {"strongBuy": "Strong Buy", "buy": "Buy", "hold": "Hold", "sell": "Sell", "strongSell": "Strong Sell"}
        if yf_rec in formatted_recs_dict:
            yf_rec = formatted_recs_dict[yf_rec]
    print(yf_rec)
    
    
    res = {
        "annual_ret": float(annual_ret), 
        "vol": float(vol), 
        "sharpe": float(sharpe), 
        "momentum": float(momentum), 
        "drawdown": float(max_drawdown),
        "yf_rec": yf_rec
    }
    
    return res, log_ret, prices

# If investment is a stock, compute and normalize its sortino, peg ratio, roe, and 52-week position score
def stock_metric(annual_ret, log_ret, roe, peg, low, high, curr_price):
    downside = log_ret[log_ret < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = (annual_ret - risk_free) / (downside_vol + 1e-8)
    sortino = np.clip(sortino / 2, -1, 1)
    
    if roe != "N/A":
        roe_score = np.clip(roe / 0.20, -1, 1)
    else:
        roe_score = 0
    
    if peg != "N/A" and peg > 0:
        peg_score = np.clip((2 - peg) / 2, -1, 1)
    else:
        peg_score = 0
        
    if high != "N/A" and low != "N/A":
        if high > low:
            pos = (curr_price - low) / (high - low)
            price_score = np.clip(1 - (2 * pos), -1, 1)
        else:
            price_score = 0
        
    return float(sortino), float(roe_score), float(peg_score), float(price_score)

# If investment is a fund/ETF, compute information ratio and return
def fund_metric(spy_ret, log_ret):
    combined = pd.concat(
        [log_ret, spy_ret],
        axis=1,
        join="inner"
    )
    combined.columns = ["fund", "spy"]
    active_ret = combined["fund"] - combined["spy"]
    tracking_error = active_ret.std() * np.sqrt(252)
    ann_excess_return = (combined["fund"].mean() - combined["spy"].mean()) * 252
    info_ratio = ann_excess_return / (tracking_error + 1e-8)

    return float(info_ratio)
    
# Get difference between benchmark S&P 500's and current investment's metrics
def get_spy_diff(curr, spy):
    sharpe_diff = np.clip((curr["sharpe"] - spy["sharpe"]) / 0.5, -1, 1)
    ret_diff = np.clip((curr["annual_ret"] - spy["annual_ret"]) / 0.05, -1, 1)
    vol_diff = np.clip((curr["vol"] - spy["vol"]) / 0.2, -1, 1)
    momentum_diff = np.clip((curr["momentum"] - spy["momentum"]) / 0.30, -1, 1)
    drawdown_diff = np.clip((spy["drawdown"] - curr["drawdown"]) / 0.3, -1, 1)
    
    return sharpe_diff, ret_diff, vol_diff, momentum_diff, drawdown_diff

# Interpretation of scores for a stock
def get_stock_score(score):
    if score > 0.6:
        return "Strong Buy"
    elif score > 0.20:
        return "Moderate Buy"
    elif score > -0.20:
        return "Hold"
    elif score > -0.60:
        return "Weak Sell"
    else:
        return "Strong Sell"
    
# Interpretation of scores for a fund/ETF
def get_fund_score(score):
    if score > 0.6:
        return "Strong Outperformer"
    elif score > 0.2:
        return "Outperformer"
    elif score > -0.2:
        return "Market-like"
    else:
        return "Underperformer"



