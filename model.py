import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.layers import TextVectorization
from keras import Input
from features import get_ticker_info

# Convert data from dictionary into vector, flatten sector vector
# Also removes growth score and value score from features to make model learn from other features
def to_vector(data):
    return [
        data['region_US'],
        data['beta'],
        data['market_cap_log'],
        data['momentum'],
        data['volatility'],
        *data['sector_vector'],
        data['turnover'],
        data['expense_ratio'],
        data['concentration']
    ]
    
def main():
    # import tickers and weights from app.py
    tickers = {'AAPL': 29.44, 'FELG': 25.25, 'FNILX': 14.03, 'FSPGX': 11.71, 'FXAIX': 12.12, 'LMT': 7.45}
    weights = [29.44, 25.25, 14.03, 11.71, 12.12, 7.45]
    
    values = [
        {'region_US': 1, 'beta': 1.109, 'market_cap_log': 28.967259522305625, 'growth_score': 9.94056948, 'value_score': 0.10059784818305519, 'momentum': 0.0002400861857521086, 'volatility': 0.21626791464998998, 'sector_vector': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'turnover': 0.0, 'expense_ratio': 0.0, 'concentration': 1.0},
        {'region_US': 1, 'beta': 1.17, 'momentum': -0.0561836, 'volatility': 0.1755, 'market_cap_log': 25.328436022934504, 'growth_score': 1.0, 'value_score': 0.0, 'sector_vector': [0.5132, 0.051799998, 0.1142, 0.1409, 0.088199995, 0.0754, 0.0057, 0.010199999, 0.0, 0.0, 0.0], 'expense_ratio': 0.0018000001, 'turnover': 0.6, 'concentration': 0.41704979799999997},
        {'region_US': 1, 'beta': 1.02, 'momentum': 0.0035926, 'volatility': 0.153, 'market_cap_log': 25.328436022934504, 'growth_score': 0.5, 'value_score': 0.5, 'sector_vector': [0.3319, 0.1221, 0.099700004, 0.1073, 0.0994, 0.086899996, 0.0354, 0.0535, 0.0199, 0.024300002, 0.0], 'expense_ratio': 0.0, 'turnover': 0.3, 'concentration': 0.2523083},
        {'region_US': 1, 'beta': 1.18, 'momentum': -0.0541259, 'volatility': 0.177, 'market_cap_log': 25.328436022934504, 'growth_score': 1.0, 'value_score': 0.0, 'sector_vector': [0.4901, 0.0575, 0.13149999, 0.1282, 0.0863, 0.060500003, 0.0042, 0.0299, 0.0038, 0.0032, 0.0], 'expense_ratio': 0.00035, 'turnover': 0.3, 'concentration': 0.411369686},
        {'region_US': 1, 'beta': 1.0, 'momentum': 0.0073911, 'volatility': 0.15, 'market_cap_log': 25.328436022934504, 'growth_score': 0.5, 'value_score': 0.5, 'sector_vector': [0.33080003, 0.122600004, 0.1011, 0.1073, 0.098400004, 0.0866, 0.0348, 0.0543, 0.019299999, 0.0249, 0.0], 'expense_ratio': 0.00014999999, 'turnover': 0.3, 'concentration': 0.25457480000000005},
        {'region_US': 1, 'beta': 0.24, 'market_cap_log': 25.71377856070457, 'growth_score': 5.5343396, 'value_score': 0.18068999945540035, 'momentum': 0.23690664280752083, 'volatility': 0.2684069625477072, 'sector_vector': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'turnover': 0.0, 'expense_ratio': 0.0, 'concentration': 1.0},
    ]
    
    # Different growth, value, and blended funds with differing size (small, mid, large) and region (foreign vs US) for training model
    funds_data = [
        'CIPMX', 'CIPSX', 'FHOFX', 'LSGRX', 'MGHRX', 'POAGX', 'PBCKX', 'PCBIX', 'PNAIX', 'VUG', 'VIGAX',
        'VONG', 'VRGWX', 'VOOG', 'VSPGX', 'VMFGX', 'VSGAX', 'VBK', 'FSPGX', 'FELG', 'VHIAX', 'EAALX', 'BUFGX',
        'FMKFX', 'FLCNX', 'FAGOX', 'AGTHX', 'IWF', 'QQQ', 'VIGIX', 'VIGRX', 'FCNKX', 'FCNTX', 'FBCCX', 'FBCEX', 
        'FBCHX', 'FBCJX', 'FBCKX', 'FBGKX', 'FBGRX', 'FDGRX', 'FGCKX', 'FOCKX','FOTDX', 'VMGAX', 'TRLGX', 'TILIX', 
        'TRIHX', 'TRIRX', 'FSBDX', 'FVWSX', 'MLRMX', 'POGBX', 'TRIWX', 'BPTIX', 'BPTRX', 'UANQX', 'AMAGX', 'CGFYX',
        'GGIPX', 'GSINX', 'VWILX', 'MQGIX', 'JAENX', 'WCMIX', 'JMGFX', 'APHIX', 'TQAAX', 'HLGEX', 'LGGAX', 'POAGX', 
        'OPOCX', 'HRSIX', 'TQMIX',
        # Value funds
        'AMRMX', 'BUSA', 'CGCV', 'CGDV', 'DHLYX', 'DFAT', 'DODGX', 'FEQIX', 'HAMVX', 'JVMIX', 'LSVQX', 'MVCJX',
        'MEIJX', 'NEOYX', 'OAKM', 'OAKMX', 'SCHD', 'ONEY', 'VHYAX', 'VYM', 'VMVAX', 'VONV', 'VMFVX', 'VSMVX', 
        'VOE', 'VRVIX', 'VSPVX', 'VOOV', 'VSIAX', 'VBR', 'VVIAX', 'VTV', 'VEVRX', 'EQWL', 'SLASX', 'VSIIX', 
        'JVMTX', 'PAIGX', 'RRIGX', 'TRIGX', 'TROZX', 'TRTIX', 'DFSVX', 'FINVX', 'TRTZX', 'DISVX', 'DFFVX', 
        'HILAX', 'HILCX', 'HILIX', 'FIWCX', 'VSCAX', 'VSMCX', 'FDVLX', 'FVLKX', 'VVOAX', 'VVOCX', 'VVOIX', 
        'VVOSX', 'SFNNX', 'FCPVX', 'FRCSX', 'CUURX', 'CSSCX', 'BINCX', 'LMNOX', 'BIAVX', 'ROFCX', 'GMCFX', 'LYRAX', 'EISAX',
        # Blend funds
        'RFNGX', 'RICGX', 'QCERX', 'FDFIX', 'FCLKX', 'BSPGX', 'DYNF', 'LRGF', 'OGFAX', 'NESNX', 'SWPPX', 'SPLG',
        'VFFSX', 'VGIAX', 'VMCTX', 'FNILX', 'FXAIX', 'ADGAX', 'FSCDX', 'PRFZ', 'ELFNX', 'VTPSX', 'VTSNX', 'VDIPX', 
        'VIMAX', 'NAESX', 'FSMDX', 'DFIEX', 'TCIWX', 'MGIAX', 'MINJX', 'MINHX', 'CGIFX', 'IGFFX', 'RIGFX', 'FEORX', 
        'SGOIX', 'DFISX', 'FTHAX', 'STESX', 'STEYX', 'VSEQX', 'FKMCX', 'FMCSX', 'BGORX', 'BROAX', 'BROIX', 'BROKX',
        'GGDPX', 'GICAX', 'GIRLX', 'TGIRX', 'TMPRX', 'MYSIX', 'BRXAX', 'CSMFX', 'SEEIX', 'MNSQX', 'OSCAX', 'OSSIX', 
        'BRMIX', 'BRMKX', 'GABSX', 'MDCIX', 'PENNX', 'PCITX', 'VDIPX', 'STESX', 'IGFFX', 'FEORX', 'SGOIX', 'IRCYX', 'PVNBX'
        ]

    y = label_category(funds_data)
    funds = get_ticker_info(funds_data)
    model = train_funds(funds, y)
    print("model classes: ", model.classes_)
    classify(tickers, weights, model)

# Sorts training dataset into growth, value, and blended funds for training/testing model based on their category from yfinance
def label_category(funds_data):
    y = []
    
    for fund in funds_data:
        var = yf.Ticker(fund)
        category = var.funds_data.fund_overview.get('categoryName', 'N/A')
        cat = category.lower()
        
        if "growth" in cat:
            y.append("growth")
        elif "value" in cat:
            y.append("value")
        elif "blend" in cat:
            y.append("blend")

    return y
    
# Train model based on over 200 growth, value, and blended funds 
def train_funds(funds, y):
    X = []
    for i in range(len(funds)):
        X.append(to_vector(funds[i]))
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2,
        random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )
    
    # Commented out code was previously used to ensure model was actually learning by shuffling data
    #y_shuffled = np.random.permutation(y_train)
    #model.fit(X_train, y_shuffled)
    
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test)) # Print test accuracy
    scores = cross_val_score(model, X, y, cv=5) # Calculate and print Cross-Validation Accuracy
    print(scores.mean())
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

# Use model on user's portfolio to estimate their portfolio's probability of growth, value, and blended stocks/funds
def classify(portfolio, weights, model):
    # Normalize weights to ensure they add up to 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Store the classification estimated by model for each stock/fund from user's portfolio
    breakdown = {0:0, 1:0, 2:0}
    features = get_ticker_info(portfolio.keys())  # Get information for each stock/fund
    print(features)
    
    port_beta = 0
    port_vol = 0
    port_momentum = 0
    port_sector = np.zeros(len(features[0]['sector_vector']))
    
    for i in range(len(weights)):
        # Flatten each stock's dictionary into a vector
        # Remove their growth and value score to ensure model actually learns
        feature_vec = to_vector(features[i]) 
        
        # Returns the probability estimates for each possible class (continuous probability) 
        # Used instead of model.predict() so each stock/fund isn't defined discretely 
        probs = model.predict_proba([feature_vec])[0] 
        for j in range(3):
            # Add to total breakdown score the probabiliy for each type * its weight in portfolio
            breakdown[j] += probs[j] * weights[i] 
        
        # Add to portfolio's total beta, volatility, momentum, and sector breakdown multiplied by element's weight
        port_beta += weights[i] * features[i]['beta']
        port_vol += weights[i] * features[i]['volatility']
        port_momentum += weights[i] * features[i]['momentum']
        port_sector += np.array(features[i]['sector_vector']) * weights[i]
    
    port_sector = port_sector.astype(float)
   
    format_output(breakdown, port_beta, port_vol, port_momentum, port_sector)

def format_output(breakdown, port_beta, port_vol, port_momentum, port_sector):
    # Print portfolio's overall beta, volatility, momentum
    print(f"Overall Beta: {port_beta:.3f}")    
    print(f"Overall Volatility: {port_vol:.3f}")  
    print(f"Overall Momentum: {port_momentum:.3f}")
    
    # Label names for the sectors from features.py to match to total sector vector 
    sector_names = ["Technology", "Financial Services", "Consumer Cyclical", "Communication Services", "Healthcare", "Industrials", 
               "Energy", "Consumer Defensive", "Basic Materials", "Utilities", "Real Estate"]
    sectors_dict = dict(zip(sector_names, port_sector.tolist()))
    print("Portfolio's Sector Breakdown: ")
    for k, v in sectors_dict.items():
        print(f"{k}: {(v*100):.2f}%")
    
    # Label breakdown dict's indices
    label_map = {
        0: "Blend", 
        1: "Growth", 
        2: "Value"
    }
    labeled_breakdown = {
        label_map[k]: float(np.squeeze(v)) for k, v in breakdown.items()
    }
    for k, v in labeled_breakdown.items():
        print(f"{k}: {(v*100):.3f}%")
    

if __name__=="__main__":
    main()