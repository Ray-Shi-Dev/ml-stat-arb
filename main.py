# main.py
import yaml
from src import data_loader, features, clustering, model_ensemble, backtest

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    print("Starting pipeline...")
    
    # 1. Load configuration
    config = load_config()
    print(f"Loaded config: {config['model']['ensemble_method']} model with {config['clustering']['n_clusters']} clusters.")
    
    # 2. Run your pipeline steps (Conceptual examples based on your files)
    
    # df = data_loader.load_data(config['data']['raw_path'])
    # df_features = features.extract_features(df)
    # clusters = clustering.run_clustering(df_features, config['clustering']['n_clusters'])
    # model = model_ensemble.train_model(df_features, config['model'])
    # results = backtest.run_backtest(model, df_features)
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()