import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from API.method.Schemas import PredictIn, PredictOut
from API.method.computeSimilarity import ComputeSimilarity
from Data_fetch_and_load import DataFetchandLoad
from EmbeddingCache import EmbeddingCache
from model_GNN import ContrastiveLoss

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

# 배치 샘플링 (실제 구현에서는 더 효율적인 방법 사용 필요)
batch=64
traveler_idx = torch.randint(0, data.num_travelers, (batch,))
trip_idx = torch.randint(data.num_travelers, data.num_travelers + data.num_trips, (batch,))
pos_dest_idx = torch.randint(data.num_travelers + data.num_trips, data.num_nodes, (batch,))
neg_dest_idx = torch.randint(data.num_travelers + data.num_trips, data.num_nodes, (batch, 5))  # 각 positive에 대해 5개의 negative

print("Model Testing")
model.eval()
with torch.no_grad():
    similarity, scores, _ = model(data.x, data.edge_index, traveler_idx, trip_idx, torch.cat([pos_dest_idx.unsqueeze(1), neg_dest_idx], dim=1))

    labels = torch.zeros_like(similarity)
    labels[:, 0] = 1  # 첫 번째 목적지가 positive sample

    cont_loss = ContrastiveLoss(similarity, labels)
    loss_fn = nn.MSELoss()

    mse_loss = loss_fn(scores[:, 0, :], data.y[pos_dest_idx])

    loss = cont_loss.loss() + mse_loss
    
print("loss :", loss)


def get_embedding():
    mlflow.artifacts.download_artifacts(
        run_id=best_run[0].info.run_id, artifact_path='embeddings', dst_path ='.'
    )
    loaded_embeddings = torch.load('embeddings/embedding_cache.pth')
    return loaded_embeddings

print("Get Embedding")
embedding_cache = EmbeddingCache()
embedding_cache.cache = get_embedding()

def find_next_dest(traveler_id, trip_id, num_recommendations=5):
    print("Dest Finding")
    model.eval()
    with torch.no_grad():
        traveler_idx = torch.tensor([traveler_id]).long()
        trip_idx = torch.tensor([data.num_travelers + trip_id]).long()
        dest_idx = torch.arange(data.num_travelers + data.num_trips, data.num_nodes).long()

        print(f"traveler_idx.size(): {traveler_idx.size()}")
        print(f"trip_idxeler_idx.size(): {trip_idx.size()}")
        print(f"dest_idx.size(): {dest_idx.size()}")
        
        similarity, scores, embedding = model(data.x, data.edge_index, traveler_idx, trip_idx, dest_idx)

        print(f"traveler_embedding: {embedding[traveler_idx]}")
        print(f"trip_embedding: {embedding[trip_idx]}")
        print(f"similarity: {similarity[:10]}")
        
        # 유사도와 예측 점수를 결합하여 최종 순위 결정
        final_scores = similarity.squeeze(0) + F.softmax(scores, dim=1)[:, 0]

        
        top_destinations = final_scores.argsort(descending=True)[:num_recommendations]
        top_scores = scores[top_destinations]
        
    return top_destinations, top_scores


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
    
    print(f"traveler_id: {traveler_id}")
    print(f"trip_id: {trip_id}")
    next_destination_id, predicted_scores = find_next_dest(traveler_id, trip_id)
    print(f"Similar traveler: {traveler_id},\n{PRE_TMA.iloc[traveler_id].squeeze()}")
    print(f"Similar trip: {trip_id}, \n{PRE_TA.iloc[trip_id].squeeze()}")

    return PredictOut(next_destination_id=next_destination_id[0].item(), 
                      predicted_rating=round(float(predicted_scores[0][0].item()), 2), 
                      predicted_recommend=round(float(predicted_scores[0][1].item()), 2), 
                      predicted_revisit=round(float(predicted_scores[0][2].item()), 2))


if __name__ == "__main__":

    input = {'traveler_id': 1,
            'GENDER': None,
            'AGE_GRP': None,
            'M': 7,
            'TRAVEL_STATUS_DESTINATION': 11,
            'TRAVEL_STYL': 0,
            'TRAVEL_MOTIVE':0,
            'trip_id': 3,
            'TRAVEL_PERIOD': 1,
            'SHOPPING': 0,
            'PARK': 0,
            'HISTORY': 0,
            'TOUR': 0,
            'SPORTS': 0,
            'ARTS': 0,
            'PLAY': 0,
            'CAMPING': 0,
            'FESTIVAL': 0,
            'SPA': 1,
            'EDUCATION': 0,
            'DRAMA': 1,
            'PILGRIMAGE': 0,
            'WELL': 1,
            'SNS': 0,
            'HOTEL': 1,
            'NEWPLACE': 0,
            'WITHPET': 0,
            'MIMIC': 0,
            'ECO': 0,
            'HIKING': 1}
    print("Predict")
    print(f"input data: {input}")
    prediction=predict(input=input)
    print(f"result: {prediction}")
    print(f"Next Dest Info: \n{PRE_VAI.iloc[prediction.next_destination_id].squeeze()}")