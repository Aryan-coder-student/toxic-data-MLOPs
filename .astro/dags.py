# airflow/dags/toxic_classifier_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys

# Add src/ to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.preprocess import main as preprocess_main
from models.train import main as train_main

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    'toxic_classifier_pipeline',
    default_args=default_args,
    description='Pipeline for toxic comment classification',
    schedule_interval=timedelta(days=1),  # Run daily (adjust as needed)
    start_date=datetime(2025, 4, 6),
    catchup=False,
) as dag:


    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_main,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_main,
    )


    deploy_task = BashOperator(
        task_id='deploy_fastapi',
        bash_command='docker build -t toxic-classifier-api . && docker run -d -p 8000:8000 toxic-classifier-api',
    )

    
    preprocess_task >> train_task >> deploy_task