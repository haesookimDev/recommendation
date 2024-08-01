import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from API.method.Schemas import PredictIn, PredictOut
from API.method.computeSimilarity import ComputeSimilarity
from Recommender.Data_fetch_and_load import DataFetchandLoad
from Recommender.EmbeddingCache import EmbeddingCache

import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow

from dotenv import load_dotenv
from argparse import ArgumentParser

print("Setting Default Configuration")
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
print("Creatin MLFlow Client")
client = mlflow.MlflowClient()

# 실험 ID 설정 (실제 실험 ID로 변경해야 함)
experiment_id = "0"

# 실험의 모든 실행 가져오기
print("Finding best run id")
runs = client.search_runs(experiment_id)

best_run = client.search_runs(
        experiment_id, order_by=["metrics.loss"], max_results=1
    )
print(best_run)
print(f"run_id: {best_run[0].info.run_id}" )

print("Loading Model from MLFlow")
model = mlflow.pytorch.load_model(f"runs:/{best_run[0].info.run_id}/{args.model_name}")

#Data Fetching
print("Data Fetching")
PATH_DATA="../DataBase/Data/Central/TL_csv/"

TMA = pd.read_csv(PATH_DATA+'tn_traveller_master_여행객_Master_A.csv')
TA = pd.read_csv(PATH_DATA+'tn_travel_여행_A.csv')
VAI = pd.read_csv(PATH_DATA+'tn_visit_area_info_방문지정보_A.csv')

dataloader = DataFetchandLoad(TMA, TA, VAI)

PRE_TMA, PRE_TA, PRE_VAI = dataloader.fetch()

traveler_len = len(PRE_TMA)

# 데이터 로드
print("Data Loading")

data = dataloader.load(PRE_TMA, PRE_TA, PRE_VAI)

def get_embedding():
    
    loaded_embeddings = torch.load('../Recommender/embeddings/embedding_cache.pth')
    return loaded_embeddings

print("Get Embedding")
embedding_cache = EmbeddingCache()
embedding_cache.cache = get_embedding()

if __name__ == "__main__":

    traveler_id = 553
    trip_id = 2681

    model.eval()
    with torch.no_grad():
        cached_embeddings = embedding_cache.get('embeddings')

        traveler_embedding = model.linear(cached_embeddings[traveler_id])
        trip_embedding = model.linear(cached_embeddings[traveler_len + trip_id])

        # 모든 여행지와의 유사도 계산
        destination_embeddings = model.linear(cached_embeddings[data.mask])
        similarities = F.cosine_similarity(traveler_embedding + trip_embedding, destination_embeddings)

        # 가장 유사한 여행지 선택
        next_destination_id = similarities.argmax().item()
        predicted_scores = model.linear(cached_embeddings[data.mask][next_destination_id])

    
    print("Predict")
    print(f"traveler_id: {traveler_id}")
    print(f"trip_id: {trip_id}")

    print(f"traveler_embedding: {traveler_embedding}")
    print(f"trip_embedding: {trip_embedding}")

    print(f"destination_embeddings: {destination_embeddings}")
    print(f"similarities: {similarities.squeeze()[:10]}")
    
    print(f"result: {next_destination_id}")