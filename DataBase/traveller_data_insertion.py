import time

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

HOST_URL = os.getenv('DB_CONTAINER_HOST')
USERNAME = os.getenv('POSTGRES_USER')
USERPASSWORD = os.getenv('POSTGRES_PASSWORD')
DBNAME = os.getenv('POSTGRES_DB')
PORT = os.getenv('PORT')

TP_CD_DICT = {0 : 'None',
              1 : 'SHOPPING', 
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

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS traveller (
        id SERIAL PRIMARY KEY,
        TRAVELER_ID CHAR(7),
        GENDER CHAR(1),
        AGE_GRP NUMERIC(3),
        TRAVEL_LIKE_SIDO_1 NUMERIC(2),
        TRAVEL_LIKE_SGG_1 NUMERIC(5),
        TRAVEL_LIKE_SIDO_2 NUMERIC(2),
        TRAVEL_LIKE_SGG_2 NUMERIC(5),
        TRAVEL_LIKE_SIDO_3 NUMERIC(2),
        TRAVEL_LIKE_SGG_3 NUMERIC(5),
        TRAVEL_STYL_1 NUMERIC(1),
        TRAVEL_STYL_2 NUMERIC(1),
        TRAVEL_STYL_3 NUMERIC(1),
        TRAVEL_STYL_4 NUMERIC(1),
        TRAVEL_STYL_5 NUMERIC(1),
        TRAVEL_STYL_6 NUMERIC(1),
        TRAVEL_STYL_7 NUMERIC(1),
        TRAVEL_STYL_8 NUMERIC(1),
        TRAVEL_STATUS_DESTINATION VARCHAR(10),
        TRAVEL_MOTIVE_1 NUMERIC(10),
        TRAVEL_MOTIVE_2 NUMERIC(10),
        TRAVEL_MOTIVE_3 NUMERIC(10)
    );
    CREATE TABLE IF NOT EXISTS travel (
        id SERIAL PRIMARY KEY,
        TRAVELER_ID CHAR(7),
        TRAVEL_ID CHAR(9),
        TRAVEL_PERIOD NUMERIC(2)
    );
    CREATE TABLE IF NOT EXISTS travel_purpose (
        id SERIAL PRIMARY KEY,
        TRAVEL_ID CHAR(9),
        SHOPPING NUMERIC(1),
        PARK NUMERIC(1),
        HISTORY NUMERIC(1),
        TOUR NUMERIC(1),
        SPORTS NUMERIC(1),
        ARTS NUMERIC(1),
        PLAY NUMERIC(1),
        CAMPING NUMERIC(1),
        FESTIVAL NUMERIC(1),
        SPA NUMERIC(1),
        EDUCATION NUMERIC(1),
        DRAMA NUMERIC(1),
        PILGRIMAGE NUMERIC(1),
        WELL NUMERIC(1),
        SNS NUMERIC(1),
        HOTEL NUMERIC(1),
        NEWPLACE NUMERIC(1),
        WITHPET NUMERIC(1),
        MIMIC NUMERIC(1),
        ECO NUMERIC(1),
        HIKING NUMERIC(1),
        None NUMERIC(1)
    );
    """
    print("create_table_query")
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

def get_data():
    TMA = pd.read_csv("Data/Central/TL_csv/tn_traveller_master_여행객_Master_A.csv")
    TA = pd.read_csv("Data/Central/TL_csv/tn_travel_여행_A.csv")

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
                   'TRAVEL_MOTIVE_3']].copy()
    
    PRE_TA = TA[['TRAVELER_ID', 'TRAVEL_ID', 'TRAVEL_PURPOSE']].copy()

    PRE_TA['TRAVEL_PERIOD'] = TMA['TRAVEL_STATUS_YMD'].apply(tripdaycheck)
    
    return PRE_TMA.fillna(0), PRE_TA.fillna(0)

def insert_data(db_connect, PRE_TMA, PRE_TA):

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

    TP_STRING = PRE_TA['TRAVEL_PURPOSE'].split(';')

    for i in TP_STRING:
        if i.isdigit():
            TP_DICT[TP_CD_DICT[int(i)]]=1
        else:
            TP_DICT['None']=1

    insert_row_query = f"""
    INSERT INTO traveller (TRAVELER_ID,
        GENDER, 
        AGE_GRP, 
        TRAVEL_LIKE_SIDO_1, 
        TRAVEL_LIKE_SGG_1, 
        TRAVEL_LIKE_SIDO_2, 
        TRAVEL_LIKE_SGG_2,
        TRAVEL_LIKE_SIDO_3,
        TRAVEL_LIKE_SGG_3,
        TRAVEL_STYL_1,
        TRAVEL_STYL_2,
        TRAVEL_STYL_3, 
        TRAVEL_STYL_4, 
        TRAVEL_STYL_5, 
        TRAVEL_STYL_6, 
        TRAVEL_STYL_7, 
        TRAVEL_STYL_8, 
        TRAVEL_STATUS_DESTINATION, 
        TRAVEL_MOTIVE_1, 
        TRAVEL_MOTIVE_2, 
        TRAVEL_MOTIVE_3)
        VALUES (
            '{PRE_TMA.TRAVELER_ID}',
            '{PRE_TMA.GENDER}',
            {int(PRE_TMA.AGE_GRP)},
            {int(PRE_TMA.TRAVEL_LIKE_SIDO_1)},
            {int(PRE_TMA.TRAVEL_LIKE_SGG_1)},
            {int(PRE_TMA.TRAVEL_LIKE_SIDO_2)},
            {int(PRE_TMA.TRAVEL_LIKE_SGG_2)},
            {int(PRE_TMA.TRAVEL_LIKE_SIDO_3)},
            {int(PRE_TMA.TRAVEL_LIKE_SGG_3)},
            {int(PRE_TMA.TRAVEL_STYL_1)},
            {int(PRE_TMA.TRAVEL_STYL_2)},
            {int(PRE_TMA.TRAVEL_STYL_3)},
            {int(PRE_TMA.TRAVEL_STYL_4)},
            {int(PRE_TMA.TRAVEL_STYL_5)},
            {int(PRE_TMA.TRAVEL_STYL_6)},
            {int(PRE_TMA.TRAVEL_STYL_7)},
            {int(PRE_TMA.TRAVEL_STYL_8)},
            '{PRE_TMA.TRAVEL_STATUS_DESTINATION}',
            {int(PRE_TMA.TRAVEL_MOTIVE_1)},
            {int(PRE_TMA.TRAVEL_MOTIVE_2)},
            {int(PRE_TMA.TRAVEL_MOTIVE_3)});
    INSERT INTO travel (TRAVELER_ID, TRAVEL_ID, TRAVEL_PERIOD)
        VALUES (
            '{PRE_TA.TRAVELER_ID}',
            '{PRE_TA.TRAVEL_ID}',
            {int(PRE_TA.TRAVEL_PERIOD)}
        );
    INSERT INTO travel_purpose (TRAVEL_ID, 
        SHOPPING, 
        PARK, 
        HISTORY,
        TOUR, 
        SPORTS, 
        ARTS, 
        PLAY,
        CAMPING, 
        FESTIVAL, 
        SPA, 
        EDUCATION, 
        DRAMA, 
        PILGRIMAGE, 
        WELL, 
        SNS, 
        HOTEL, 
        NEWPLACE, 
        WITHPET, 
        MIMIC, 
        ECO, 
        HIKING,
        None)
        VALUES (
            '{PRE_TA.TRAVELER_ID}',
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
            {TP_DICT['None']}
        );
        """
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()
        print("insert_row_query")

def generate_data(db_connect, PRE_TMA, PRE_TA):
    while True:
        insert_data(db_connect, PRE_TMA.sample(1).squeeze(), PRE_TA.sample(1).squeeze())
        time.sleep(1)

if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user=USERNAME, 
        password=USERPASSWORD,
        host=HOST_URL,
        port=PORT,
        database=DBNAME,
    )

    create_table(db_connect)
    TMA, TA = get_data()
    PRE_TMA, PRE_TA = preprocessing(TMA, TA)

    generate_data(db_connect, PRE_TMA, PRE_TA)