import pandas as pd

def preprocess_data():
    input_path = '/opt/airflow/data/vehicles.csv'
    output_path = '/opt/airflow/data/processed.csv'

    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to {output_path}")