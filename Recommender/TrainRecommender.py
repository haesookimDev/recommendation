import pandas as pd
import numpy as np
from datetime import datetime
import os, sys

from sklearn.preprocessing import LabelEncoder

from model_GNN import TravelRecommendationGNN
from Data_processing import DataProcessing

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Preprocessing.PreprocessingTravelLog import PreprocessingRawData

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchinfo import summary
import mlflow


from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')

device = "cuda" if torch.cuda.is_available() else "cpu"

#Data Fetching
print("Data Fetching")
PATH_DATA="../DataBase/Data/Central/TL_csv/"

TMA = pd.read_csv(PATH_DATA+'tn_traveller_master_여행객_Master_A.csv')
TA = pd.read_csv(PATH_DATA+'tn_travel_여행_A.csv')
VAI = pd.read_csv(PATH_DATA+'tn_visit_area_info_방문지정보_A.csv')

preprocessingRawData = PreprocessingRawData()

PRE_TMA, PRE_TA, PRE_VAI = preprocessingRawData.preprocessing(TMA, TA, VAI)

PRE_TMA.reset_index(drop=True, inplace=True)
PRE_TA.reset_index(drop=True, inplace=True)
PRE_VAI.reset_index(drop=True, inplace=True)

data_processing = DataProcessing()

# 데이터 로드
print("Data Loading")
travelers = PRE_TMA  # 여행자 데이터
trips = PRE_TA  # 여행 데이터
destinations = PRE_VAI  # 여행지 데이터

data = data_processing.prepare_data(travelers, trips, destinations)

#모델 하이퍼파라미터
print("Prepare Model")
num_features = data.x.size(1)
hidden_channels = 64
num_classes = 3  # 평점, 추천 점수, 재방문 의향 점수
epochs = 300

model = TravelRecommendationGNN(num_features, hidden_channels, num_classes)

loss_fn = nn.MSELoss()
metric_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 모델 학습 함수
def train_model(model, data, loss_fn, metrics_fn, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)


        loss = loss_fn(out[data.mask], data.y[data.mask])
        accuracy = metrics_fn(out[data.mask], data.y[data.mask])

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            mlflow.log_metric("loss", f"{loss:3f}", step=((epoch + 1) // 10))
            mlflow.log_metric("accuracy", f"{accuracy:3f}", step=((epoch + 1) // 10))
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, Accuracy: {accuracy:3f}')

print("Start Train Model")

signature = mlflow.models.signature.infer_signature(model_input=data.x.detach().numpy(), model_output=model(data.x, data.edge_index).detach().numpy())
input_sample = data.x[:10].detach().numpy()

with mlflow.start_run():
    params = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
        "signature": signature
    }
    # Log training parameters.
    mlflow.log_params(params)
    
    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt", artifact_path="model_summary")

    train_model(model, data, loss_fn, metric_fn, optimizer, epochs)

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "model", signature=signature, input_example=input_sample)

