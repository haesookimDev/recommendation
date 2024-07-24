import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from API.method.Schemas import PredictIn, PredictOut
from API.method.computeSimilarity import ComputeSimilarity
from Recommender.Data_fetch_and_load import DataFetchandLoad

#Data Fetching
print("Data Fetching")
PATH_DATA="../DataBase/Data/Central/TL_csv/"

TMA = pd.read_csv(PATH_DATA+'tn_traveller_master_여행객_Master_A.csv')
TA = pd.read_csv(PATH_DATA+'tn_travel_여행_A.csv')
VAI = pd.read_csv(PATH_DATA+'tn_visit_area_info_방문지정보_A.csv')

dataloader = DataFetchandLoad(TMA, TA, VAI)

PRE_TMA, PRE_TA, PRE_VAI = dataloader.fetch()

def test(input: dict):
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

    return traveler_id, trip_id


if __name__ == "__main__":

    input = {'traveler_id': None,
            'GENDER': None,
            'AGE_GRP': None,
            'M': 7,
            'TRAVEL_STATUS_DESTINATION': 41,
            'TRAVEL_STYL': 7,
            'TRAVEL_MOTIVE':5,
            'trip_id': None,
            'TRAVEL_PERIOD': 2,
            'SHOPPING': 0,
            'PARK': 0,
            'HISTORY': 1,
            'TOUR': 1,
            'SPORTS': 1,
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
            'HOTEL': 0,
            'NEWPLACE': 1,
            'WITHPET': 0,
            'MIMIC': 0,
            'ECO': 0,
            'HIKING': 0}
    print("Predict")
    print(f"input data: {input}")
    traveler_id, trip_id=test(input=input)
    print(f"result: { traveler_id, trip_id}")
    print(f"confirm traveler: { PRE_TMA.iloc[traveler_id].squeeze() }")
    print(f"confirm trip: { PRE_TA.iloc[trip_id].squeeze() }")

