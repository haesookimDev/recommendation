import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from API.Schemas import PredictIn, PredictOut
from API.computeSimilarity import ComputeSimilarity
from Data_fetch_and_load import DataFetchandLoad
from EmbeddingCache import EmbeddingCache

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
model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/{args.model_name}")

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

print("Model Testing")
model.eval()
with torch.no_grad():
    _, out = model(data.x, data.edge_index)

    loss_fn = nn.MSELoss()

    loss = loss_fn(out[data.mask], data.y[data.mask])
    
print("loss :", loss)


def get_embedding():
    mlflow.artifacts.download_artifacts(
        run_id=best_run.info.run_id, artifact_path='embeddings', dst_path ='.'
    )
    loaded_embeddings = torch.load('embeddings/embedding_cache.pth')    
    return loaded_embeddings

print("Get Embedding")
embedding_cache = EmbeddingCache()
embedding_cache.cache = get_embedding()

def find_next_dest(traveler_id, trip_id):
    print("Dest Finding")
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
    return next_destination_id, predicted_scores


def predict(input: dict) -> PredictOut:
    df = pd.DataFrame([input])
    traveler_id = df['traveler_id'].squeeze()
    trip_id = df['trip_id'].squeeze()
    similarities = ComputeSimilarity(df=df)
    if traveler_id==None:
        print("Calculate Traveler Similarity")
        traveler_id = similarities.content_based_similarity_traveler(PRE_TMA)
    if trip_id == None:
        print("Calculate Trip Similarity")
        trip_id = similarities.content_based_similarity_trip(PRE_TA)
    
    next_destination_id, predicted_scores = find_next_dest(traveler_id, trip_id)
    print(f"Similar traveler: {traveler_id},\n{PRE_TMA.iloc[traveler_id].squeeze()}")
    print(f"Similar trip: {trip_id}, \n{PRE_TA.iloc[trip_id].squeeze()}")

    return PredictOut(next_destination_id=next_destination_id, 
                      predicted_rating=round(int(predicted_scores[0]), 2), 
                      predicted_recommend=round(int(predicted_scores[1]), 2), 
                      predicted_revisit=round(int(predicted_scores[2]), 2))


if __name__ == "__main__":

    input = {'traveler_id': None,
            'GENDER': None,
            'AGE_GRP': None,
            'M': 7,
            'TRAVEL_STATUS_DESTINATION': 11,
            'TRAVEL_STYL': 6,
            'TRAVEL_MOTIVE':1,
            'trip_id': None,
            'TRAVEL_PERIOD': 2,
            'SHOPPING': 1,
            'PARK': 1,
            'HISTORY': 0,
            'TOUR': 1,
            'SPORTS': 0,
            'ARTS': 0,
            'PLAY': 0,
            'CAMPING': 0,
            'FESTIVAL': 0,
            'SPA': 1,
            'EDUCATION': 0,
            'DRAMA': 0,
            'PILGRIMAGE': 0,
            'WELL': 0,
            'SNS': 0,
            'HOTEL': 1,
            'NEWPLACE': 0,
            'WITHPET': 0,
            'MIMIC': 0,
            'ECO': 0,
            'HIKING': 0}
    print("Predict")
    print(f"input data: {input}")
    prediction=predict(input=input)
    print(f"result: {prediction}")
    print(f"Next Dest Info: \n{PRE_VAI.iloc[prediction.next_destination_id].squeeze()}")