import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_and_load_data():
    DATASET = "austinreese/craigslist-carstrucks-data"
    DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
    os.makedirs(DIR, exist_ok=True)

    CSV_FILENAME = "vehicles.csv"
    ZIP_FILENAME = "craigslist-carstrucks-data.zip"

    csv_path = os.path.join(DIR, CSV_FILENAME)
    zip_path = os.path.join(DIR, ZIP_FILENAME)

    if os.path.exists(csv_path):
        print(f"'{CSV_FILENAME}' already exists. Skipping download.")
    else:
        api = KaggleApi()
        api.authenticate()

        print("Downloading dataset from Kaggle...")
        api.dataset_download_files(DATASET, path=DIR, unzip=False)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DIR)

        print("Download and extraction complete.")

    # โหลด DataFrame
    df = pd.read_csv(csv_path)

    columns_to_keep = [
        'price', 'year', 'manufacturer', 'model', 'condition', 'cylinders',
        'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'size',
        'type', 'paint_color', 'state'
    ]

    df = df[columns_to_keep]

    SAVE_FILENAME = 'cleaned_data.csv'
    save_path = os.path.join(DIR, SAVE_FILENAME)
    df.to_csv(save_path,index=False)
