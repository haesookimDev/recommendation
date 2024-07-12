import pandas as pd
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

traveler_len = len(PRE_TMA)

# 데이터 로드
print("Data Loading")

data = dataloader.load(PRE_TMA, PRE_TA, PRE_VAI)

model.eval()
with torch.no_grad():
    _, out = model(data.x, data.edge_index)

    loss_fn = nn.MSELoss()

    loss = loss_fn(out[data.mask], data.y[data.mask])
    
print("loss :", loss)


def get_embedding():
    artifact_uri = mlflow.get_artifact_uri('embeddings/embedding_cache.pth')
    loaded_embeddings = torch.load(artifact_uri.replace('file://', ''))    
    return loaded_embeddings

embedding_cache = EmbeddingCache()
embedding_cache.cache = get_embedding()


def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    traveler_id = df['traveler_id']
    trip_id = df['trip_id']
    similarities = ComputeSimilarity(df=df)
    if traveler_id==None:
        traveler_id = similarities.content_based_similarity_traveler(PRE_TMA)
    if df['trip_id'] != None:
        trip_id = similarities.content_based_similarity_trip(PRE_TA)
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
        
    return PredictOut(next_destination_id=next_destination_id, predicted_rating=predicted_scores[0], predicted_recommend=predicted_scores[1], predicted_revisit=predicted_scores[2])


