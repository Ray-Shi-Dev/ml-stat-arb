import numpy as np
from data_loader import fetch_reit_data
from clustering import find_optimal_clusters, find_cointegrated_pairs
from features import engineer_features
from model_ensemble import train_and_evaluate_pair
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 1. Fetch liquid US REITs 
    tickers =
    print("Fetching REIT data from Yahoo Finance...")
    prices, returns = fetch_reit_data(tickers, '2015-01-01', '2023-01-01')
    
    # 2. Cluster & Pair
    print("Running K-Means Clustering and ADF Cointegration Tests...")
    clusters = find_optimal_clusters(returns)
    pairs = find_cointegrated_pairs(prices, clusters)
    print(f"Found {len(pairs)} cointegrated pairs.")
    
    # 3. Model First Valid Pair
    if pairs:
        best_pair = pairs
        print(f"Modeling Pair: {best_pair['Asset1']} & {best_pair['Asset2']}")
        
        print("Generating Causal Savitzky-Golay features...")
        data_df = engineer_features(best_pair)
        
        print("Training Ensemble models (XGBoost, LightGBM, CatBoost)...")
        results, auc_score = train_and_evaluate_pair(data_df)
        
        if results is not None:
            print(f"Out-of-Sample ROC-AUC: {auc_score:.4f}")
            
            results = (results - results) / results
            results = np.where(abs(results) > 2, 1, 0)
            results = np.where((abs(results) > 2) & (results < 0.70), 1, 0)
            
            baseline_trades = results.sum()
            ml_trades = results.sum()
            vetoed = baseline_trades - ml_trades
            
            print(f"Baseline proposed {baseline_trades} trades.")
            print(f"ML Overlay vetoed {vetoed} doomed trades.")