import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import os

# Set up MLflow tracking - using Docker container
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("used_car_price_prediction")

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_and_tune_model(model, param_grid, X_train, X_test, y_train, y_test, model_name):
    with mlflow.start_run(run_name=f"{model_name}_tuning") as run:
        # Log the Docker environment
        mlflow.set_tag("mlflow.source.type", "LOCAL")
        mlflow.set_tag("mlflow.source.name", os.path.basename(__file__))
        mlflow.set_tag("mlflow.docker.container_id", os.getenv('HOSTNAME', 'local'))
        
        # Create scorer for GridSearchCV
        scorer = make_scorer(rmse_score, greater_is_better=False)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=2
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=scorer)
        
        # Log parameters
        mlflow.log_params(best_params)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("cv_rmse_mean", -cv_scores.mean())
        mlflow.log_metric("cv_rmse_std", cv_scores.std())
        
        # Log model with appropriate flavor
        if model_name == "xgboost":
            mlflow.xgboost.log_model(best_model, "model")
        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")
        
        # Log feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save feature importance plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.xticks(rotation=45)
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()
        
        return best_model, best_params, rmse, r2

if __name__ == "__main__":
    # Load your data
    # Replace this with your actual data loading code
    # df = pd.read_csv('your_data.csv')
    
    # Example data preparation (modify according to your actual features)
    # X = df.drop('price', axis=1)
    # y = df['price']
    
    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # Model configurations
    models = {
        "xgboost": {
            "model": xgb.XGBRegressor(random_state=42),
            "param_grid": {
                "max_depth": [3, 6, 8],
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        },
        "random_forest": {
            "model": RandomForestRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "lightgbm": {
            "model": lgb.LGBMRegressor(random_state=42),
            "param_grid": {
                "num_leaves": [31, 50, 70],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "subsample": [0.8, 0.9, 1.0]
            }
        }
    }
    
    # Train and evaluate all models
    results = {}
    
    # Uncomment and modify the following code when your data is ready
    # for model_name, config in models.items():
    #     print(f"\nTraining {model_name}...")
    #     best_model, best_params, rmse, r2 = train_and_tune_model(
    #         model=config["model"],
    #         param_grid=config["param_grid"],
    #         X_train=X_train_scaled,
    #         X_test=X_test_scaled,
    #         y_train=y_train,
    #         y_test=y_test,
    #         model_name=model_name
    #     )
    #     
    #     results[model_name] = {
    #         "best_params": best_params,
    #         "rmse": rmse,
    #         "r2": r2
    #     }
    #     
    #     print(f"\n{model_name.upper()} Results:")
    #     print(f"Best Parameters: {best_params}")
    #     print(f"RMSE: {rmse:.4f}")
    #     print(f"R2 Score: {r2:.4f}\n")
    #     print("-" * 80) 