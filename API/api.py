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

MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()

@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    
    MODEL.eval()
    with torch.no_grad():
        out = MODEL(data.x, data.edge_index)
        traveler_embedding = out[traveler_id]
        trip_embedding = out[len(travelers) + trip_id]

        # 모든 여행지와의 유사도 계산
        destination_embeddings = out[data.mask]
        similarities = F.cosine_similarity(traveler_embedding + trip_embedding, destination_embeddings)

        # 가장 유사한 여행지 선택
        next_destination_id = similarities.argmax().item()
        predicted_scores = out[data.mask][next_destination_id]
        
    pred = MODEL.predict(df).item()
    return PredictOut(next_destination_id=pred)