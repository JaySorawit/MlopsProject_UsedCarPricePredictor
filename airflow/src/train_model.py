import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os 

def train_model():
    input_name = 'preprocessed_data.npz'
    output_name = 'model.pkl'
    # output_path = '/opt/airflow/data/processed.csv'

    DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
    input_path = os.path.join(DIR, input_name)
    output_path = os.path.join(DIR, output_name)
    print(f"Loading preprocessed data from {input_path}")
    data = np.load(input_path, allow_pickle=True)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"✅ Model trained:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE:  {mae:.2f}")
    print(f"   R²:   {r2:.2f}")

    # Save model
    joblib.dump(model, output_path)
    print(f"📁 Model saved to: {output_path}")

    return output_path
