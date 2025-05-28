# MLflow Experiment Tracking

This directory contains the setup for MLflow experiment tracking for the Used Car Price Predictor project.

## Models Included

The experiment includes three different models:
1. XGBoost
2. Random Forest
3. LightGBM

Each model undergoes parameter tuning using GridSearchCV with 5-fold cross-validation.

## Setup and Usage

1. Install additional requirements:
```bash
pip install xgboost lightgbm scikit-learn mlflow pandas numpy
```

2. Start the MLflow tracking server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

3. The tracking server will be available at `http://localhost:5000`

4. Run experiments using the provided script:
```bash
python train_with_mlflow.py
```

## What's Being Tracked

### Models and Parameters

#### XGBoost
- max_depth: [3, 6, 8]
- n_estimators: [100, 200, 300]
- learning_rate: [0.01, 0.05, 0.1]
- subsample: [0.8, 0.9, 1.0]
- colsample_bytree: [0.8, 0.9, 1.0]

#### Random Forest
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, 30]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

#### LightGBM
- num_leaves: [31, 50, 70]
- max_depth: [3, 5, 7]
- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [100, 200, 300]
- subsample: [0.8, 0.9, 1.0]

### Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation RMSE (mean and std)

### Artifacts
- Best trained model for each algorithm
- Parameter tuning results
- Cross-validation scores

## Viewing Results

1. Open your web browser and navigate to `http://localhost:5000`
2. You can:
   - Compare different models and their performances
   - View metrics and parameters for each experiment
   - Download saved models
   - Visualize metric trends across different models
   - Compare cross-validation results

## Best Practices

1. Always use meaningful run names (automatically set to model name)
2. Compare models based on cross-validation scores
3. Consider the trade-off between model performance and complexity
4. Document any data preprocessing steps
5. Use tags to organize experiments by model type

## Customization

You can modify `train_with_mlflow.py` to:
- Add more models
- Adjust parameter grids for tuning
- Implement different cross-validation strategies
- Add custom metrics
- Modify preprocessing steps 