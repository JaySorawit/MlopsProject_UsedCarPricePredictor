from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# เพิ่ม path ให้ Airflow รู้จักโค้ดใน src/
sys.path.append('/opt/airflow/src')

# Import functions จากไฟล์ Python
from preprocess_data import preprocess_data
from train_model import train_model

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='used_car_price_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='ML pipeline for used car price prediction using Airflow + MLflow',
    tags=['ml', 'airflow', 'mlflow']
) as dag:

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    preprocess_task >> train_task
