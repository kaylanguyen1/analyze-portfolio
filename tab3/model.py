import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .features import get_ticker_info
    
def get_portfolio_classification(tickers, weights, features):
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
    beta, vol, momentum, sectors, breakdown = classify(features, weights, model)
    
    return beta, vol, momentum, sectors, breakdown

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
    #print(classification_report(y_test, y_pred))
    
    return model

# Use model on user's portfolio to estimate their portfolio's probability of growth, value, and blended stocks/funds
def classify(features, weights, model):
    # Normalize weights to ensure they add up to 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    # Store the classification estimated by model for each stock/fund from user's portfolio
    breakdown = {0:0, 1:0, 2:0}

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
    
    return port_beta, port_vol, port_momentum, port_sector, breakdown
