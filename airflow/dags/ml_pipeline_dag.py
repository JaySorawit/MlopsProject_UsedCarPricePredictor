from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import sys

# เพิ่ม path ให้ Airflow รู้จักโค้ดใน src/
sys.path.append('/opt/airflow/src')

# Import ฟังก์ชันแต่ละ step
from data_collection import data_collection
from data_cleaning import data_cleaning
from preprocess_data import preprocess_data

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

    # จุดเริ่มต้น / จุดสิ้นสุด
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # Data Collection
    data_collection_task = PythonOperator(
        task_id='data_collection',
        python_callable=data_collection
    )

    # Data Cleaning
    data_cleaning_task = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning
    )

    # Preprocessing (บันทึก .npz)
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    # DAG pipeline structure
    start >> data_collection_task >> data_cleaning_task >> preprocess_task >>  end
