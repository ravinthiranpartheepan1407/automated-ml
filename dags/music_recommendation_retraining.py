import sys
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Add the directory containing train_and_deploy.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_and_deploy import load_and_preprocess_data
from train_and_deploy import train_models
from train_and_deploy import log_best_model
from train_and_deploy import recommend_songs
from train_and_deploy import main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 9, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'music_recommendation_retraining',
    default_args=default_args,
    description='A DAG to retrain and redeploy the music recommendation model',
    schedule_interval=timedelta(days=1),
)

preprocess = PythonOperator(
    task_id='Data_Preprocessing',
    python_callable=load_and_preprocess_data,
    provide_context=True,
    dag=dag,
)

train = PythonOperator(
    task_id='Model_Training',
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)

log_model = PythonOperator(
    task_id='Log_model',
    python_callable=log_best_model,
    provide_context=True,
    dag=dag,
)

recommend = PythonOperator(
    task_id='Recommendation',
    python_callable=recommend_songs,
    provide_context=True,
    dag=dag,
)

output = PythonOperator(
    task_id='Output',
    python_callable=main,
    provide_context=True,
    dag=dag,
)

preprocess >> train >> log_model >> recommend >> output