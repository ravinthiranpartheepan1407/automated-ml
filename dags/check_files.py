import os
import pandas
from pathlib import Path

print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir())
print("Contents of /opt/airflow:", os.listdir("/opt/airflow"))
print("Contents of /opt/airflow/datasets:", os.listdir("/opt/airflow/datasets"))

file_path = "/opt/airflow/datasets/tracks.csv"
print("Does file exist?", Path(file_path).exists())

df = pd.read_csv(file_path)

