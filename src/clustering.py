import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def find_optimal_clusters(returns_df, max_k=15):
    """Find optimal K using KneeLocator on standardized returns."""
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_df.T) 
    
    inertias =
    ks = range(2, max_k + 1)
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(returns_scaled)
        inertias.append(kmeans.inertia_)
        
    kn = KneeLocator(list(ks), inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee if kn.knee else 5
    
    kmeans_best = KMeans(n_clusters=optimal_k, random_state=42).fit(returns_scaled)
    cluster_map = pd.Series(kmeans_best.labels_, index=returns_df.columns)
    return cluster_map

def find_cointegrated_pairs(prices_df, cluster_map):
    """Test pairs within the same cluster for cointegration (ADF test)."""
    pairs =
    clusters = cluster_map.unique()
    
    for c in clusters:
        assets = cluster_map[cluster_map == c].index.tolist()
        if len(assets) < 2: continue
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                y = np.log(prices_df[asset1])
                x = sm.add_constant(np.log(prices_df[asset2]))
                model = sm.OLS(y, x).fit()
                beta = model.params.iloc
                
                spread = y - (beta * np.log(prices_df[asset2]))
                adf_stat = adfuller(spread) 
                
                if adf_stat < 0.05: 
                    pairs.append({'Asset1': asset1, 'Asset2': asset2, 'Beta': beta, 'Spread': spread})
                    
    return pairs