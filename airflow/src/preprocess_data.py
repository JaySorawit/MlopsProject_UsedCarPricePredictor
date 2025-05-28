import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import time
import joblib
import pandas as pd
import numpy as np

def preprocess_data():
    input_path = '/opt/airflow/data/cleaned_data.csv'
    output_path = '/opt/airflow/data/preprocessed_data.npz'

    # Load dataset
    df = pd.read_csv(input_path)
    X = df.drop('price', axis=1)
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Define columns
    ordinal_col = ['condition']
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ordinal_col]

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    condition_order = ['Unknown', 'salvage', 'fair', 'good', 'excellent', 'like new', 'new']
    train_conditions = X_train['condition'].unique()
    final_condition_order = [c for c in condition_order if c in train_conditions]

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ordinal', OrdinalEncoder(categories=[final_condition_order],
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column transformer
    transformers_list = []
    if numeric_cols:
        transformers_list.append(('num', numeric_transformer, numeric_cols))
    if ordinal_col:
        transformers_list.append(('ord', ordinal_transformer, ordinal_col))
    if categorical_cols:
        transformers_list.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop'
    )

    # Fit and transform
    start_time = time.time()
    preprocessor.fit(X_train, y_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    preprocess_time = time.time() - start_time

    print(f"✅ Preprocessing completed in {preprocess_time:.2f} seconds.")
    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}")

    # Try to get feature names
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        print("✅ Feature names extracted from pipeline.")
    except Exception as e:
        print(f"⚠️ Warning: Could not get detailed feature names: {e}")
        num_output_features = X_train_processed.shape[1]
        feature_names_out = [f'feature_{i}' for i in range(num_output_features)]

    # Save as .npz
    np.savez(output_path,
             X_train=X_train_processed,
             X_test=X_test_processed,
             y_train=y_train.to_numpy(),
             y_test=y_test.to_numpy(),
             feature_names=feature_names_out)

    print(f"📁 Saved preprocessed data to: {output_path}")

    # Save preprocessor
    preprocessor_path = '/opt/airflow/models/preprocessor.joblib'
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Saved preprocessor to: {preprocessor_path}")
    return output_path

