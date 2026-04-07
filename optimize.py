import numpy as np
import scipy.optimize as sco
import streamlit as st
import pandas as pd

# Function to find optimal weights 
def optimize(returns_df, portfolio, initial_val, weights, max_alloc):
    expected_ret = returns_df.mean() * 252 # Expected annual return of each asset
    cov_mtx = returns_df.cov() * 252 # Covariance matrix: how volatile each asset is and how they move together
    
    frontier_data = efficient_frontier(
        expected_ret,
        cov_mtx,
        portfolio,
        returns_df,
        weights, 
        initial_val,
        max_alloc
    )
    
    return frontier_data
    
def calc_efficient_frontier(expected_ret, cov_mtx, portfolio, max_alloc, target_ret):
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, # Ensure weights sum up to 1 for a fully invested portfolio
        {'type': 'ineq', 'fun': lambda w: w @ expected_ret - target_ret}
    ] 
    bounds = tuple((0,max_alloc) for _ in range(len(portfolio))) # Cap for each stock's allocation defined by user input on slider to enforce diversification
    init_guess = np.array(len(portfolio)*[1./len(portfolio)]) #Starting point for optimizer
    
    result = sco.minimize(portfolio_volatility, init_guess, args=(cov_mtx,), method='SLSQP', bounds=bounds, constraints=constraints) 
    
    return result.x

def efficient_frontier(expected_ret, cov_mtx, portfolio, returns_df, weights, initial_val, max_alloc, num_points=5):
    current = current_metrics(returns_df, weights, initial_val)
    market_ret = expected_ret.mean()
    expected_ret_clean = 0.7 * expected_ret + 0.3 * market_ret
    avg_ret = expected_ret_clean.mean()
    ret_std = expected_ret_clean.std()
    current_ret = current[3]
    min_ret = max(avg_ret - ret_std, current_ret - 0.2)
    max_ret = avg_ret + 2 * ret_std
    target_ret = np.linspace(min_ret, max_ret, num_points)
    frontier_data = []
    
    for ret in target_ret:
        try:
            w = calc_efficient_frontier(expected_ret_clean, cov_mtx, portfolio, max_alloc, ret) # Portfolio allocation for each stock
            portfolio_ret = (returns_df * w).sum(axis=1)
            portfolio_val = (1 + portfolio_ret).cumprod() * initial_val
            final_val = portfolio_val.iloc[-1] # Final portfolio value starting with its initial value
            total_ret = final_val / initial_val - 1 # Actual change in return over entire time period
            vol = portfolio_ret.std() * np.sqrt(252) # Annualized standard deviation of returns to measure portfolio's risk
            return_diff = total_ret - current_ret # Difference in return of portfolio with optimized weights vs. current portfolio
            sharpe = total_ret / vol if vol != 0 else 0
            
            frontier_data.append({
                "target_return": ret,
                "volatility": vol,
                "final_value": final_val,
                "total_return": total_ret,
                "return_diff": return_diff,
                "weights": w,
                "sharpe": sharpe
            })
            
        except Exception as e:
            print(f"Optimization failed for target {ret}: {e}")
            continue
    return frontier_data

def portfolio_volatility(weights, cov_mtx):
    return np.sqrt(weights.T @ cov_mtx @ weights)

def current_metrics(returns_df, weights, initial_val):
    weights = pd.Series(weights, index=returns_df.columns)
    current_ret = (returns_df * weights).sum(axis=1)
    current_val = (1 + current_ret).cumprod() * initial_val
    current_final = current_val.iloc[-1]
    current_total_ret = current_final / initial_val - 1
    current_vol = current_ret.std() * np.sqrt(252)
    metrics = [current_ret, current_val, current_final, current_total_ret, current_vol]
    
    return metrics
    
#Return computed metrics for optimal and current portfolio
def compare(returns_df, weights, initial_value):
    # Daily portfolio returns
    current_returns = (returns_df * weights).sum(axis=1)
    current_value = (1 + current_returns).cumprod() * initial_value
    current_final = current_value.iloc[-1]
    current_return = current_final / initial_value - 1
    current_vol = current_returns.std() * np.sqrt(252)
    current_sharpe = current_return / current_vol if current_vol != 0 else 0
    
    metrics = {
        "initial_value": initial_value,
        "current_final": current_final,
        "current_return": current_return,
        "current_vol": current_vol,
        "current_sharpe": current_sharpe
    }
    
    return metrics

def compare_exp(metrics):
    opt_final = round(metrics["optimized_final"], 2)
    cur_final = round(metrics["current_final"], 2)
    value_diff = round((metrics["optimized_final"] - metrics["current_final"]), 2)
    return_diff = round(((metrics["optimized_return"] - metrics["current_return"]) * 100), 2)
    opt_vol = round((metrics["optimized_vol"]*100), 2)
    cur_vol = round((metrics["current_vol"]*100), 2)
    
    return opt_final, cur_final, value_diff, return_diff, opt_vol, cur_vol