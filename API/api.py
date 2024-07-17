import os
from dotenv import load_dotenv

from DownloadModel import download_model
from Recommender.Data_fetch_and_load import DataFetchandLoad
from Recommender.EmbeddingCache import EmbeddingCache
from API.method.computeSimilarity import ComputeSimilarity

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import pandas as pd
from fastapi import FastAPI
from API.method.Schemas import PredictIn, PredictOut

print("Setting Default Configuration")
load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')


client = mlflow.tracking.MlflowClient()
experiment_id = "0"
best_run = client.search_runs(
    experiment_id, order_by=["metrics.loss"], max_results=1
)[0]

def get_model():
    
    model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/model")
    return model

def get_embedding():
    mlflow.artifacts.download_artifacts(
        run_id=best_run.info.run_id, artifact_path='embeddings', dst_path ='.'
    )
    artifact_uri = mlflow.get_artifact_uri('embeddings/embedding_cache.pth')
    loaded_embeddings = torch.load(artifact_uri.replace('file://', ''))    
    return loaded_embeddings

MODEL = get_model()

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

traveler_len = len(PRE_TMA)

print("Model Testing")
MODEL.eval()
with torch.no_grad():
    _, out = MODEL(data.x, data.edge_index)

    loss_fn = nn.MSELoss()

    loss = loss_fn(out[data.mask], data.y[data.mask])
    
print("loss :", loss)

embedding_cache = EmbeddingCache()
embedding_cache.cache = get_embedding()

def find_next_dest(traveler_id, trip_id):
    print("Dest Finding")
    MODEL.eval()
    with torch.no_grad():
        cached_embeddings = embedding_cache.get('embeddings')

        traveler_embedding = MODEL.linear(cached_embeddings[traveler_id])
        trip_embedding = MODEL.linear(cached_embeddings[traveler_len + trip_id])

        # 모든 여행지와의 유사도 계산
        destination_embeddings = MODEL.linear(cached_embeddings[data.mask])
        similarities = F.cosine_similarity(traveler_embedding + trip_embedding, destination_embeddings)

        # 가장 유사한 여행지 선택
        next_destination_id = similarities.argmax().item()
        predicted_scores = MODEL.linear(cached_embeddings[data.mask][next_destination_id])
    return next_destination_id, predicted_scores

# Create a FastAPI instance
app = FastAPI()

@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    traveler_id = df['traveler_id']
    trip_id = df['trip_id']
    similarities = ComputeSimilarity(df=df)
    if traveler_id==None:
        traveler_id = similarities.content_based_similarity_traveler(PRE_TMA)
    if df['trip_id'] != None:
        trip_id = similarities.content_based_similarity_trip(PRE_TA)
    
    next_destination_id, predicted_scores = find_next_dest(traveler_id, trip_id)
    print(f"Similar traveler: {traveler_id},\n{PRE_TMA.iloc[traveler_id].squeeze()}")
    print(f"Similar trip: {trip_id}, \n{PRE_TA.iloc[trip_id].squeeze()}")
        
    return PredictOut(next_destination_id=next_destination_id, predicted_rating=predicted_scores[0], predicted_recommend=predicted_scores[1], predicted_revisit=predicted_scores[2])