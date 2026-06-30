# Portfolio Analysis & Optimization Platform

A full-stack portfolio analytics platform built with Python, Streamlit, yfinance, scikit-learn, TensorFlow/Keras, and SciPy that enables users to analyze, optimize, classify, and evaluate investment portfolios. The application combines quantitative finance, machine learning, and interactive data visualization to provide portfolio optimization, investment style classification, volatility forecasting, and data-driven stock recommendations through a unified web interface.

# Highlights

- Implemented **Modern Portfolio Theory** using **SciPy's Sequential Least Squares Programming (SLSQP)** optimizer to generate constrained efficient frontiers with customizable asset allocation limits

- Developed two machine learning pipelines consisting of a **Random Forest classifier** for portfolio style prediction and a **TensorFlow/Keras LSTM neural network** for forecasting future portfolio volatility from historical return sequences

- Engineered **quantitative financial features** from market data, including Sharpe ratio, Sortino ratio, Information ratio, momentum, maximum drawdown, beta, valuation metrics, sector exposure, and concentration statistics

- Created an interactive **Streamlit** dashboard featuring real-time visualizations, dynamic portfolio analytics, and responsive user controls powered by **Plotly**

# Features

## Portfolio Overview

- Upload portfolio holdings via CSV and automatically calculate:

    - Current portfolio value
    - Position weights
    - Gains/losses
    - Historical cumulative portfolio performance

- Interactive visualizations of portfolio value and individual holdings over time

- Detailed price and return tables generated directly from historical market data

## Portfolio Optimization

- Modern Portfolio Theory (MPT) optimization using constrained nonlinear optimization

- User-adjustable maximum allocation constraints for individual assets

- Efficient frontier generation with multiple optimal portfolios

- Portfolio comparison including:

    - Expected return
    - Volatility
    - Sharpe ratio
    - Optimal asset weights

- Actionable buy/sell recommendations converted directly into share quantities based on optimization results

## Portfolio Classification

- Automatic investment style classification using a supervised machine learning model

- Random Forest classifier trained on 200+ ETFs spanning:
    - Growth
    - Value
    - Blend

- Feature engineering for both equities and funds using financial, sector, and risk characteristics

- Portfolio-wide calculations including:

    - Beta
    - Momentum
    - Annual volatility
    - Sector allocation

- Long Short-Term Memory (LSTM) neural network for forecasting next month's portfolio volatility

## Investment Recommendations

- Compare up to four investments simultaneously against the S&P 500

- Compute quantitative investment scores using custom financial metrics

- Generate Buy/Hold/Sell recommendations for stocks and performance classifications for ETFs

- Display analyst recommendation summaries from Yahoo Finance alongside custom model predictions

# Project Structure

`analyze-portfolio/`

│

├── `.streamlit/`

│   └── `config.toml`

│

├── `tab3`/

│   ├── `__init___.py`

│   ├── `features.py`

│   ├── `model.py`

│   ├── `risk_model.py`

│

├── `tab4`/

│   ├── `__init__.py`

│   └── `ticker_analysis.py`

│

├── `app.py`

├── `holdings.csv`

├── `metrics.py`

├── `optimize.py` 

├── `recs.py`

├── `requirements.txt`

├── `runtime.txt`

└── `README.md`


# File Overview

## `app.py`

Main Streamlit application responsible for:

- Application layout

- Navigation between tabs

- User interaction

- Visualization generation

- Data orchestration across all modules

## `metrics.py`

Calculates portfolio performance metrics including:

- Current position value

- Portfolio weights

- Unrealized gains/losses

- Historical cumulative portfolio performance

## `optimize.py`

Implements portfolio optimization using Modern Portfolio Theory.

Major responsibilities include:

- Expected return estimation

- Covariance matrix construction

- Efficient frontier generation

- Optimal asset allocation using constrained optimization

## `recs.py`

Generates actionable portfolio rebalancing recommendations by comparing current holdings against optimized allocations and converting allocation differences into recommended share transactions.

## `tab3/`

### `features.py`

Builds machine learning feature vectors for both equities and ETFs.

Engineered features include:

- Beta

- Momentum

- Volatility

- Market capitalization

- Growth/value scores

- Sector vectors

- Turnover

- Expense ratio

- Portfolio concentration

### `model.py`

Trains and evaluates the Random Forest portfolio classification model.

Responsibilities include:

- Dataset preprocessing

- Label encoding

- Feature engineering

- Cross-validation

- Portfolio style prediction

- Aggregate portfolio characteristic computation

### `risk_model.py`

Implements the LSTM volatility forecasting pipeline.

Responsibilities include:

- Historical return preprocessing

- Sequence generation

- Feature scaling

- Neural network training

- Future volatility prediction

- Portfolio covariance reconstruction

## `tab4/`

### `ticker_analysis.py`

Provides investment analysis and recommendation functionality for user-input investments.

Calculates:

- Annual return

- Volatility

- Sharpe ratio

- Sortino ratio

- Information ratio

- Momentum

- Maximum drawdown

- Relative S&P 500 performance

- Composite investment score

- Buy/Hold/Sell recommendation

## `holdings.csv`

Example portfolio file demonstrating required upload format. 

Required columns:

- Ticker

- Shares

- Total Cost Basis

- Acquisition Date

## `requirements.txt` and `runtime.txt`

Python dependencies required for local execution/Streamlit deployment and specified Python runtime version used during Streamlit Community Cloud deployment.

# Access

Application can be accessed using this [link:] ( https://kaylanguyen1-analyze-portfolio-app-ve70xn.streamlit.app/)

# Installation Instructions

1. Clone repository or download ZIP file

```bash
https://github.com/kaylanguyen1/analyze-portfolio.git
```

2. Access folder containing code

```bash
cd analyze-portfolio
``` 

3. Install Python packages

```bash
pip install -r requirements.txt
```

4. Run application

```bash
streamlit run app.py
```

# Complications

- I had trouble ensuring accuracy and realistic recommendations throughout, so I added checks with yfinance and other sources throughout when possible

- Recommendations for user-input stocks are not as predictive as I was hoping for due to added complexity in methods and analysis

- Models run slower than I hoped, causing stalls in the application starting up tabs 3 and 4

# Future Improvements

- Factor in multiples of the same stock/ETF – currently, input doesn't accomodate more than one of the same stock/ETF due to calculation errors for portfolio weights, etc.

- Add Monte Carlo portfolio simulations and Value at Risk (VaR) analysis

- Cache market data and trained models to reduce API calls and improve performance

- Integrate additional fundamental indicators such as earnings growth, analyst estimate revisions, and free cash flow for enhanced recommendation models

- Add more forward-looking evaluation and other more complex features for analyzing stocks

# Sources

I referenced information, vocabulary, and concepts from sources such as the University of Washington, Yahoo Finance, Morningstar, Fidelity, Vanguard, and Charles Schwab. The recommendations, analysis, and calculations may not be perfect; this was purely an attempt to learn more about investing and to provide a possible resource for those who may be starting out in investing as well!

# License

This project is intended for educational and portfolio demonstration purposes. Market data is provided through Yahoo Finance via the yfinance library and should not be interpreted as financial advice.
