import pandas as pd
import numpy as np
from datetime import datetime
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Data_fetch_and_load import DataFetchandLoad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchinfo import summary
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
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
parser.add_argument("--run-id", dest="run_id", type=str)
args = parser.parse_args()

model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/{args.model_name}")



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
    out = model(data.x, data.edge_index)

    loss_fn = nn.MSELoss()

    loss = loss_fn(out[data.mask], data.y[data.mask])

print("loss :", loss)