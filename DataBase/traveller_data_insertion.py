import time

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

HOST_URL = os.environ.get('HOST_URL')
USERNAME = os.environ.get('USERNAME')
USERPASSWORD = os.environ.get('USERPASSWORD')
DBNAME = os.environ.get('DBNAME')
PORT = os.environ.get('PORT')

def get_data():
    TMA = pd.read_csv("./Data/Central/TL_csv/tn_traveller_master_여행객 Master_A.csv")
    TA = pd.read_csv("./Data/Central/TL_csv/tn_travel_여행_A.csv")

    return TMA, TA

def tripdaycheck(x: str)->int:
    x_1 = x.split('~')
    start = x_1[0].split('-')
    end = x_1[1].split('-')
    start = datetime(int(start[0]), int(start[1]), int(start[2]))
    end = datetime(int(end[0]), int(end[1]), int(end[2]))
    diff = end-start

    return diff.days+1

def preprocessing(TMA, TA):
    TP_DICT = {'TRAVEL_ID': '', 
               'SHOPPING': 0, 
               'PARK': 0, 
               'HISTORY': 0, 
               'TOUR': 0, 
               'SPORTS': 0, 
               'ARTS': 0, 
               'PLAY': 0,
               'CAMPING': 0, 
               'FESTIVAL': 0, 
               'SPA': 0, 
               'EDUCATION': 0, 
               'DRAMA': 0, 
               'PILGRIMAGE': 0, 
               'WELL': 0, 
               'SNS': 0, 
               'HOTEL': 0, 
               'NEWPLACE': 0, 
               'WITHPET': 0, 
               'MIMIC': 0, 
               'ECO': 0, 
               'HIKING': 0}
    
    TP_CD_DICT = {1 : 'SHOPPING', 
                  2 : 'PARK', 
                  3 : 'HISTORY', 
                  4 : 'TOUR', 
                  5 : 'SPORTS', 
                  6 : 'ARTS', 
                  7 : 'PLAY',
                  8 : 'CAMPING', 
                  9 : 'FESTIVAL', 
                  10 : 'SPA', 
                  11 : 'EDUCATION', 
                  12 : 'DRAMA', 
                  13 : 'PILGRIMAGE', 
                  21 : 'WELL', 
                  22 : 'SNS', 
                  23 : 'HOTEL', 
                  24 : 'NEWPLACE', 
                  25 : 'WITHPET', 
                  26 : 'MIMIC', 
                  27 : 'ECO', 
                  28 : 'HIKING'}
    

    PRE_TMA = TMA[['TRAVELER_ID',
                   'GENDER',
                   'AGE_GRP',
                   'TRAVEL_LIKE_SIDO_1',
                   'TRAVEL_LIKE_SGG_1',
                   'TRAVEL_LIKE_SIDO_2',
                   'TRAVEL_LIKE_SGG_2',
                   'TRAVEL_LIKE_SIDO_3',
                   'TRAVEL_LIKE_SGG_3',
                   'TRAVEL_STYL_1',
                   'TRAVEL_STYL_2',
                   'TRAVEL_STYL_3', 
                   'TRAVEL_STYL_4', 
                   'TRAVEL_STYL_5', 
                   'TRAVEL_STYL_6', 
                   'TRAVEL_STYL_7', 
                   'TRAVEL_STYL_8', 
                   'TRAVEL_STATUS_DESTINATION', 
                   'TRAVEL_MOTIVE_1', 
                   'TRAVEL_MOTIVE_2', 
                   'TRAVEL_MOTIVE_3']]
    
    PRE_TA = TA[['TRAVELER_ID', 'TRAVEL_ID', 'AGE_GRP']]

    PRE_TA['TRAVEL_PERIOD'] = TMA['TRAVEL_STATUS_YMD'].apply(tripdaycheck)

    TP_STRING = TA['TRAVEL_PURPOSE']
    TP_STRING = TP_STRING.split(';')
    TP_DICT['TRAVEL_ID'] = PRE_TA['TRAVEL_ID']

    for i in TP_STRING:
        if i.isdigit():
            TP_DICT[TP_CD_DICT[int(i)]]=1
    
    return PRE_TMA, PRE_TA, TP_DICT

def insert_data(db_connect, PRE_TMA, PRE_TA, TP_DICT):

    insert_row_query = f"""
    INSERT INTO traveller
        ('TRAVELER_ID', 
        'GENDER', 
        'AGE_GRP', 
        'TRAVEL_LIKE_SIDO_1',
        'TRAVEL_LIKE_SGG_1',
        'TRAVEL_LIKE_SIDO_2',
        'TRAVEL_LIKE_SGG_2',
        'TRAVEL_LIKE_SIDO_3',
        'TRAVEL_LIKE_SGG_3',
        'TRAVEL_STYL_1',
        'TRAVEL_STYL_2',
        'TRAVEL_STYL_3', 
        'TRAVEL_STYL_4', 
        'TRAVEL_STYL_5', 
        'TRAVEL_STYL_6', 
        'TRAVEL_STYL_7', 
        'TRAVEL_STYL_8', 
        'TRAVEL_STATUS_DESTINATION', 
        'TRAVEL_MOTIVE_1', 
        'TRAVEL_MOTIVE_2', 
        'TRAVEL_MOTIVE_3')
        VALUES (
            {PRE_TMA.TRAVELER_ID},
            {PRE_TMA.GENDER},
            {PRE_TMA.TRAVEL_LIKE_SIDO_1},
            {PRE_TMA.TRAVEL_LIKE_SGG_1},
            {PRE_TMA.TRAVEL_LIKE_SIDO_2},
            {PRE_TMA.TRAVEL_LIKE_SGG_2},
            {PRE_TMA.TRAVEL_LIKE_SIDO_3},
            {PRE_TMA.TRAVEL_LIKE_SGG_3},
            {PRE_TMA.TRAVEL_STYL_1},
            {PRE_TMA.TRAVEL_STYL_2},
            {PRE_TMA.TRAVEL_STYL_3},
            {PRE_TMA.TRAVEL_STYL_4},
            {PRE_TMA.TRAVEL_STYL_5},
            {PRE_TMA.TRAVEL_STYL_6},
            {PRE_TMA.TRAVEL_STYL_7},
            {PRE_TMA.TRAVEL_STYL_8},
            {PRE_TMA.TRAVEL_STATUS_DESTINATION},
            {PRE_TMA.TRAVEL_MOTIVE_1},
            {PRE_TMA.TRAVEL_MOTIVE_2},
            {PRE_TMA.TRAVEL_MOTIVE_3},
        );
    INSERT INTO travel
        ('TRAVELER_ID', 'TRAVEL_ID', 'AGE_GRP', 'TRAVEL_PERIOD')
        VALUES (
            {PRE_TA.TRAVELER_ID},
            {PRE_TA.TRAVEL_ID},
            {PRE_TA.AGE_GRP},
            {PRE_TA.TRAVEL_PERIOD},
        );
    INSERT INTO travel
        ('TRAVEL_ID', 
        'SHOPPING', 
        'PARK', 
        'HISTORY',
        'TOUR', 
        'SPORTS', 
        'ARTS', 
        'PLAY',
        'CAMPING', 
        'FESTIVAL', 
        'SPA', 
        'EDUCATION', 
        'DRAMA', 
        'PILGRIMAGE', 
        'WELL', 
        'SNS', 
        'HOTEL', 
        'NEWPLACE', 
        'WITHPET', 
        'MIMIC', 
        'ECO', 
        'HIKING')
        VALUES (
            {TP_DICT['TRAVELER_ID']},
            {TP_DICT['SHOPPING']},
            {TP_DICT['PARK']},
            {TP_DICT['HISTORY']},
            {TP_DICT['TOUR']},
            {TP_DICT['SPORTS']},
            {TP_DICT['ARTS']},
            {TP_DICT['PLAY']},
            {TP_DICT['CAMPING']},
            {TP_DICT['FESTIVAL']},
            {TP_DICT['SPA']},
            {TP_DICT['EDUCATION']},
            {TP_DICT['DRAMA']},
            {TP_DICT['PILGRIMAGE']},
            {TP_DICT['WELL']},
            {TP_DICT['SNS']},
            {TP_DICT['HOTEL']},
            {TP_DICT['NEWPLACE']},
            {TP_DICT['WITHPET']},
            {TP_DICT['MIMIC']},
            {TP_DICT['ECO']},
            {TP_DICT['HIKING']},
        );  
        """
    print("insert_row_query")
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()

def generate_data(db_connect, df):
    while True:
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)

if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user=USERNAME, 
        password=USERPASSWORD,
        host=HOST_URL,
        port=PORT,
        database=DBNAME,
    )
    TMA, TA = get_data()
    PRE_TMA, PRE_TA, TP_DICT = preprocessing(TMA, TA)

    generate_data(db_connect, PRE_TMA, PRE_TA, TP_DICT)