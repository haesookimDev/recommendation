import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Recommender.Data_fetch_and_load import DataFetchandLoad
from Recommender.EmbeddingCache import EmbeddingCache
from computeSimilarity import ComputeSimilarity

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import pandas as pd
from fastapi import FastAPI
from schemas import PredictIn, PredictOut

def get_model():
    model = mlflow.sklearn.load_model(model_uri="./model")
    return model

def get_embedding():
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

embedding_cache = EmbeddingCache()
embedding_cache.cache = get_embedding()

# Create a FastAPI instance
app = FastAPI()

@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    similarities = ComputeSimilarity(df=df)
    traveler_id, trip_id = similarities.content_based_similarity(PRE_TA, PRE_TMA)
    MODEL.eval()
    with torch.no_grad():
        cached_embeddings = embedding_cache.get('embeddings')

        traveler_embedding = MODEL.linear(cached_embeddings[traveler_id])    
        trip_embedding = MODEL.linear(cached_embeddings[len(PRE_TMA) + trip_id])

        # 모든 여행지와의 유사도 계산
        destination_embeddings = MODEL.linear(cached_embeddings[data.mask])
        similarities = F.cosine_similarity(traveler_embedding + trip_embedding, destination_embeddings)

        # 가장 유사한 여행지 선택
        next_destination_id = similarities.argmax().item()
        predicted_scores = MODEL.linear(cached_embeddings[data.mask][next_destination_id])
        
    return PredictOut(next_destination_id=next_destination_id, predicted_rating=predicted_scores[0], predicted_recommend=predicted_scores[1], predicted_revisit=predicted_scores[2])