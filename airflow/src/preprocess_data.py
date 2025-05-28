import pandas as pd
import os

def preprocess_data():
    input_name = 'cleaned_data.csv'
    # output_path = '/opt/airflow/data/processed.csv'
    DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
    input_path = os.path.join(DIR, input_name)
    df = pd.read_csv(input_path)
    
    # ตัวอย่าง: ลบคอลัมน์ที่ไม่จำเป็น
    df = df.dropna()
    df = df.select_dtypes(include=['number'])  # เอาเฉพาะข้อมูลตัวเลข
    df.to_csv(output_path, index=False)

    # print(f"Preprocessed data saved to {output_path}")

preprocess_data()