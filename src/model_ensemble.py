import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

def train_and_evaluate_pair(df):
    """Walk-forward modeling with downsampling class equalization."""
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    features =
    
    neg = train[train['Anomaly'] == 0]
    pos = train[train['Anomaly'] == 1]
    
    if len(pos) < 10: 
        return None, np.nan 
        
    negs1 = neg.sample(n=len(pos), random_state=10)
    negs2 = neg.sample(n=len(pos), random_state=20)
    train_eq = pd.concat([negs1, pos, negs2, pos], axis=0)
    
    X_train = train_eq[features]
    y_train = train_eq['Anomaly']
    X_test = test[features]
    y_test = test['Anomaly']
    
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    lgb = LGBMClassifier(n_estimators=100, random_state=42)
    cat = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    
    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    
    pred_xgb = xgb.predict_proba(X_test)[:, 1]
    pred_lgb = lgb.predict_proba(X_test)[:, 1]
    pred_cat = cat.predict_proba(X_test)[:, 1]
    
    test = (pred_xgb + pred_lgb + pred_cat) / 3
    
    try:
        auc = roc_auc_score(y_test, test)
    except ValueError:
        auc = np.nan 
        
    return test, auc