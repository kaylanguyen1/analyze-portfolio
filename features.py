import pandas as pd
import yfinance as yf
import numpy as np

# DELETE AFTER TESTING
def create_portfolio(upload_file):
    portfolio = pd.read_csv(upload_file)
    portfolio["avg_cost"] = portfolio["total_cost_basis"] / portfolio["shares"]
    portfolio["acquisition_date"] = pd.to_datetime(portfolio["acquisition_date"])
    portfolio["acquisition_date"] = portfolio["acquisition_date"].dt.date
    
    return portfolio

def get_ticker_info(tickers):
    all_info = []
    for ticker in tickers:
        info_dict = {}
        info_dict['ticker'] = ticker
        var = yf.Ticker(ticker)
        info = var.info
        
        if info['quoteType'] == 'EQUITY':
            features = get_stock_info(ticker, info, info_dict)
        else:
            features = get_fund_info(ticker, var, info, info_dict)
        all_info.append(features)
        #print(ticker)
        #print(features)
        
    return all_info


def get_stock_info(ticker, ticker_info, info_dict):
    info_dict['region'] = ticker_info.get('country', 'N/A')
    info_dict['sector'] = ticker_info.get('sector', 'N/A')
    info_dict['beta'] = ticker_info.get('beta', 'N/A')
    info_dict['trailingPE'] = ticker_info.get('trailingPE', 'N/A')
    info_dict['marketCap'] = ticker_info.get('marketCap', 'N/A')
    info_dict['priceToBook'] = ticker_info.get('priceToBook', 'N/A')

    return standardize_stocks(ticker, info_dict)
    
def standardize_stocks(ticker, info_dict):
    features = {}
    features['region_US'] = standard_region(info_dict)
    features['beta'] = info_dict.get('beta')
    
    if info_dict['marketCap'] != 'N/A':
        marketCap = float(np.log(info_dict['marketCap']))
    
    pe = info_dict['trailingPE']
    pb = info_dict['priceToBook']
    if pe and pb:
        growth = (pe / 25) + (pb / 5)
        value = 1 / (growth + 1e-6)
        
    features['market_cap_log'] = marketCap
    features['growth_score'] = growth
    features['value_score'] = value
    momentum, vol = compute_mo_vol(ticker)
    features['momentum'] = momentum
    features['volatility'] = vol
    features['sector_vector'] = standard_sectors(info_dict)
    features['turnover'] = 0.0
    features['expense_ratio'] = 0.0
    features['concentration'] = 1.0
    
    return features

def get_fund_info(name, ticker, ticker_info, info_dict):
    category_name = ticker.funds_data.fund_overview.get('categoryName', 'N/A')
    if "Foreign" in category_name:
        info_dict['region'] = "Foreign"
    else:
        info_dict['region'] = "United States"
    info_dict['category'] = category_name
    info_dict['sector'] = ticker.funds_data.sector_weightings
    info_dict['beta'] = ticker_info.get('beta3Year', 'N/A')
    info_dict['momentum'] = ticker_info.get('trailingThreeMonthReturns', 'N/A')
    
    holds_dict = {}
    top_holdings = ticker.funds_data.top_holdings
    indexes = top_holdings.index.tolist()
    for idx in indexes:
        row = top_holdings.loc[idx]
        holds_dict[idx] = float(row['Holding Percent'])
    
    ops_dict = {}
    fund_ops = ticker.funds_data.fund_operations
    indexes = fund_ops.index.tolist()
    for idx in indexes:
        row = fund_ops.loc[idx]
        if isinstance(row[name], np.float64) or isinstance(row[name], float):
            ops_dict[idx] = float(row[name])
        else:
            ops_dict[idx] = "N/A"
    
    info_dict['concentration'] = holds_dict
    info_dict['turnover'] = ops_dict['Annual Holdings Turnover']
    if info_dict['turnover'] == 'N/A':
        info_dict['turnover'] = 0.3
    info_dict['expense_ratio'] = ops_dict['Annual Report Expense Ratio']
    if info_dict['expense_ratio'] == 'N/A':
        info_dict['expense_ratio'] = 0.3
    
    return standardize_funds(info_dict)
    
def standardize_funds(info_dict):
    features = {}
    
    features['region_US'] = standard_region(info_dict)
    features['beta'] = info_dict.get('beta')
    if info_dict['momentum'] != 'N/A':
        info_dict['momentum'] = info_dict['momentum'] / 100
    features['momentum'] = info_dict['momentum']
    features['volatility'] = info_dict.get('beta') * 0.15
    cat, growth, value = fund_mkt_growth_value(info_dict)
    features['market_cap_log'] = cat
    features['growth_score'] = growth
    features['value_score'] = value
    features['sector_vector'] = standard_sectors(info_dict)
    features['expense_ratio'] = info_dict.get('expense_ratio')
    features['turnover'] = info_dict.get('turnover')
    
    weights = list(info_dict['concentration'].values())
    weights.sort(reverse=True)
    features['concentration'] = sum(weights[:5])
    
    return features
    
def standard_region(info_dict):
    # Convert region to numeric: 1 if it's US else 0
    if info_dict['region'] == "United States" or info_dict['region'] == 'US':
        return int(1)
    else:
        return int(0)

def compute_mo_vol(ticker):
    data = yf.download(ticker, period="6mo")
    
    if len(data) < 60:
        return 0.0, 0.0

    price_now = data["Close"].iloc[-1]
    price_past = data["Close"].iloc[-60]

    price_now = float(price_now.iloc[0])
    price_past = float(price_past.iloc[0])
    momentum = (price_now / price_past) - 1
    
    returns = data["Close"].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    vol = float(vol.iloc[0])
    
    return momentum, vol

def standard_sectors(info_dict):
    sectors = ["technology", "financial_services", "consumer_cyclical", "communication_services", "healthcare", "industrials", 
               "energy", "consumer_defensive", "basic_materials", "utilities", "real_estate"]
    
    sec_vec = [0.0] * len(sectors)
    sector = info_dict['sector']
    if isinstance(sector, dict):
        for i, s in enumerate(sectors):
            sec_vec[i] = sector.get(s, 0.0)
    else:
        sector = sector.lower()
        sec_vec[sectors.index(sector)] = 1.0
        
    return sec_vec

def fund_mkt_growth_value(info_dict):
    if info_dict['category'] != 'N/A':
        c = info_dict['category'].lower()
        if "large" in c:
            cat = np.log(1e11)
        elif "mid" in c:
            cat = np.log(1e10)
        else:
            cat = np.log(1e9)
        
        if "growth" in c:
            growth = 1.0
            value = 0.0
        elif "value" in c:
            growth = 0.0
            value = 1.0
        else:
            growth, value = 0.5, 0.5
    else:
        cat = np.log(1e10)
        growth, value = 0.5, 0.5
    
    return float(cat), growth, value
    

def main():
    upload = "holdings.csv"
    portfolio = create_portfolio(upload)       
    get_ticker_info(portfolio['ticker'])
    
    

if __name__=="__main__":
    main()
