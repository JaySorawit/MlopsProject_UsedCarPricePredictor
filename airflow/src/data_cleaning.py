import pandas as pd
import matplotlib.pyplot as plt
def data_cleaning():
    # Set up visualization style
    output_path = '/opt/airflow/data/processed.csv'
    # Load dataset
    df2 = pd.read_csv(output_path)
    df = df2.copy()

    ## drop columns not uses
    drop_cols = ['id', 'url', 'region_url', 'image_url', 'description',
                    'posting_date', 'VIN', 'lat', 'long', 'county', 'size']
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore', inplace= True)
    print(f"Dropped initial columns. Shape: {df.shape}")

    ## Drop Rows Least Null 
    initial_rows = df.shape[0]
    cols_to_check_nan = ['year', 'odometer', 'manufacturer', 'model']
    cols_to_check_nan = [col for col in cols_to_check_nan if col in df.columns] # Ensure cols exist
    df.dropna(subset=cols_to_check_nan, inplace=True)
    rows_after_drop = df.shape[0]
    print(f"Dropped {initial_rows - rows_after_drop} rows with NaNs in key columns.")
    print(f"Shape after NaN row drop: {df.shape}")

    ## assign unknown safely
    for col in ['cylinders', 'condition', 'drive', 'paint_color', 'type', 'fuel', 'transmission', 'title_status']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
        else:
            print(f"⚠️ Warning: Column '{col}' not found in dataset.")


    ## Drop duplicate 
    df.drop_duplicates(inplace=True)

    ## Cardinality Reduction
    def reduce_cardinality(df, column, n):
        if column not in df.columns:
            print(f"Column {column} not found ")
            return df
        value_counts = df[column].value_counts()
        top_categories = value_counts.nlargest(n).index.tolist()
        df[column] = df[column].apply(lambda x: x if x in top_categories else 'others')
        return df

    df = reduce_cardinality(df, 'manufacturer', 30)
    df = reduce_cardinality(df, 'region', 100)

    ## clean outlier
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')

    df.dropna(subset=['price', 'year', 'odometer'], inplace=True)
    df['year'] = df['year'].astype(int)
    df['odometer'] = df['odometer'].astype(int)

    ## Price whisker
    price_percentile25 = df['price'].quantile(0.25)
    price_percentile75 = df['price'].quantile(0.75)
    iqr = price_percentile75 - price_percentile25
    price_upper_limit = price_percentile75 + 1.5 * iqr
    price_lower_limit = df['price'].quantile(0.15) # Using 15th percentile as lower bound
    df_filtered = df[(df['price'] < price_upper_limit) & (df['price'] > price_lower_limit)]

    ## odometer whisker
    odo_percentile75 = df_filtered['odometer'].quantile(0.75)
    odo_percentile25 = df_filtered['odometer'].quantile(0.25)
    odo_iqr = odo_percentile75 - odo_percentile25
    odo_upper_limit = odo_percentile75 + 1.5 * odo_iqr
    odo_lower_limit = df_filtered['odometer'].quantile(0.05) # Using 5th percentile as lower bound
    df_filtered = df_filtered[(df_filtered['odometer'] < odo_upper_limit) & (df_filtered['odometer'] > odo_lower_limit)]

    ## fillter yrar sensible
    df_filtered = df_filtered[(df['year'] >= 1995) & (df['year'] <= 2021)]
    df_filtered.shape

    ## change years to carage
    current_year = pd.Timestamp.now().year
    df_filtered['car_age'] = current_year - df_filtered['year']
    df_filtered.drop(['year'], axis=1, inplace=True)

    ## final data frame
    df_final = df_filtered.copy()
    df_final.to_csv('/opt/airflow/data/cleaned.csv', index=False)
