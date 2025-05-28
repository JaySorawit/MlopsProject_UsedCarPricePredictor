from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.operators.empty import EmptyOperator
import sys

# เพิ่ม path ให้ Airflow รู้จักโค้ดใน src/
sys.path.append('/opt/airflow/src')

# Import functions จากไฟล์ Python
from preprocess_data import preprocess_data
from data_cleaning import data_cleaning

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

    start = EmptyOperator(task_id='start')

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    data_cleaning_task = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning
    )
    end = EmptyOperator(task_id='end')

    start >> preprocess_task >> data_cleaning_task  >> end
