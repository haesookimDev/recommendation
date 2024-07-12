import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Data_fetch_and_load import DataFetchandLoad

import torch
import torch.nn as nn
import mlflow

from dotenv import load_dotenv
from argparse import ArgumentParser

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="model")
args = parser.parse_args()

# MLflow 클라이언트 생성
client = mlflow.MlflowClient()

# 실험 ID 설정 (실제 실험 ID로 변경해야 함)
experiment_id = "0"

# 실험의 모든 실행 가져오기
runs = client.search_runs(experiment_id)

best_run = client.search_runs(
        experiment_id, order_by=["metrics.loss"], max_results=1
    )[0]

print(f"run_id: {best_run.info.run_id}" )

model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/{args.model_name}")

#Data Fetching
print("Data Fetching")
PATH_DATA="../DataBase/Data/Central/TL_csv/"

TMA = pd.read_csv(PATH_DATA+'tn_traveller_master_여행객_Master_A.csv')
TA = pd.read_csv(PATH_DATA+'tn_travel_여행_A.csv')
VAI = pd.read_csv(PATH_DATA+'tn_visit_area_info_방문지정보_A.csv')

dataloader = DataFetchandLoad(TMA, TA, VAI)

PRE_TMA, PRE_TA, PRE_VAI = dataloader.fetch()

# 데이터 로드
print("Data Loading")

data = dataloader.load(PRE_TMA, PRE_TA, PRE_VAI)

model.eval()
with torch.no_grad():
    _, out = model(data.x, data.edge_index)

    loss_fn = nn.MSELoss()

    loss = loss_fn(out[data.mask], data.y[data.mask])
    
print("loss :", loss)