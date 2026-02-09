from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.services.pipeline_task import rf_qaqc_task

with DAG(
    dag_id="rf_qaqc_pipeline",
    start_date=datetime(2026, 2, 1),
    schedule_interval="@hourly",
    catchup=False,
) as dag:

    def run_qaqc(**context):
        input_data = context["ti"].xcom_pull(task_ids="fetch_pass_group")
        return rf_qaqc_task(input_data)

    qaqc = PythonOperator(
        task_id="rf_qaqc",
        python_callable=run_qaqc,
        provide_context=True,
    )
