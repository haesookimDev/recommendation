import time

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

HOST_URL = os.environ.get('HOST_URL')
USERNAME = os.environ.get('USERNAME')
USERPASSWORD = os.environ.get('USERPASSWORD')
DBNAME = os.environ.get('DBNAME')
PORT = os.environ.get('PORT')

def get_data():
    VAI = pd.read_csv("./Central/TL_csv/tn_visit_area_info_방문지정보_A.csv")

    return VAI

def split_YMD(x: str)->int:
    x_1 = x['VISIT_START_YMD'].split('~')
    YMD = x_1[0].split('-')

    return int(YMD[0]), int(YMD[1]), int(YMD[2])


def preprocessing(VAI):
    
    PRE_VAI=VAI['YMD'].apply(split_YMD)
    
    return PRE_VAI

def insert_data(db_connect, PRE_VAI):

    insert_row_query = f"""
    INSERT INTO traveller
        ('TRAVELER_ID', 
        VISIT_ORDER,
        VISIT_AREA_ID,
        VISIT_AREA_NM,
        VISIT_START_Y,
        VISIT_START_M,
        VISIT_START_D,
        ROAD_NM_ADDR,
        LOTNO_ADDR,
        X_COORD,
        Y_COORD,
        VISIT_AREA_TYPE_CD,
        VISIT_CHC_REASON_CD,
        DGSTFN,
        REVISIT_INTENTION,
        RCMDTN_INTENTION,)
        VALUES (
            {PRE_VAI.TRAVELER_ID},
            {PRE_VAI.VISIT_ORDER},
            {PRE_VAI.VISIT_AREA_ID},
            {PRE_VAI.VISIT_AREA_NM},
            {PRE_VAI.YMD[0]},
            {PRE_VAI.YMD[1]},
            {PRE_VAI.YMD[2]},
            {PRE_VAI.ROAD_NM_ADDR},
            {PRE_VAI.LOTNO_ADDR},
            {PRE_VAI.X_COORD},
            {PRE_VAI.Y_COORD},
            {PRE_VAI.VISIT_AREA_TYPE_CD},
            {PRE_VAI.VISIT_CHC_REASON_CD},
            {PRE_VAI.DGSTFN},
            {PRE_VAI.REVISIT_INTENTION},
            {PRE_VAI.RCMDTN_INTENTION}
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
    VAI = get_data()

    PRE_VAI = preprocessing(VAI)

    generate_data(db_connect, PRE_VAI)