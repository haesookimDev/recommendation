import time

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

HOST_URL = os.getenv('DB_CONTAINER_HOST')
USERNAME = os.getenv('POSTGRES_USER')
USERPASSWORD = os.getenv('POSTGRES_PASSWORD')
DBNAME = os.getenv('POSTGRES_DB')
PORT = os.getenv('PORT')

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS visit_area_info (
        id SERIAL PRIMARY KEY,
        TRAVEL_ID CHAR(9),
        VISIT_ORDER NUMERIC(2),
        VISIT_AREA_ID NUMERIC(10),
        VISIT_AREA_NM text,
        VISIT_START_Y NUMERIC(4),
        VISIT_START_M NUMERIC(2),
        VISIT_START_D NUMERIC(2),
        ROAD_NM_ADDR text,
        LOTNO_ADDR text,
        S_ADDR text,
        G_ADDR text,
        X_COORD NUMERIC(9,6),
        Y_COORD NUMERIC(8,6),
        VISIT_AREA_TYPE_CD NUMERIC(2),
        VISIT_CHC_REASON_CD NUMERIC(2),
        DGSTFN NUMERIC(1),
        REVISIT_INTENTION NUMERIC(1),
        RCMDTN_INTENTION NUMERIC(1)
    );
    """
    print("create_table_query")
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

def get_data():
    VAI = pd.read_csv("Data/Central/TL_csv/tn_visit_area_info_방문지정보_A.csv")
    
    return VAI

#날짜를 연/월/일로 분리
def split_YMD(x: str)->int:
    YMD = x.split('-')

    return int(YMD[0]), int(YMD[1]), int(YMD[2])

#주소에서 시/도, 군/구만 추출
def split_sg(x: str)->str:
    if x !=None:
        sgg = x.split(" ")
    else:
        sgg=["",""]

    #시도, 군/구 제거 ex> 강남구 > 강남
    return sgg[0], sgg[1][:-1]


def preprocessing(VAI):

    PRE_VAI = VAI.copy()
    
    PRE_VAI['YMD']=VAI['VISIT_START_YMD'].apply(split_YMD)

    PRE_VAI['SG']=VAI['LOTNO_ADDR'].fillna(" ").apply(split_sg)

    
    return PRE_VAI.fillna(0)

def insert_data(db_connect, PRE_VAI):

    insert_row_query = f"""
    INSERT INTO visit_area_info (TRAVEL_ID, 
    VISIT_ORDER,
        VISIT_AREA_ID,
        VISIT_AREA_NM,
        VISIT_START_Y,
        VISIT_START_M,
        VISIT_START_D,
        ROAD_NM_ADDR,
        LOTNO_ADDR,
        S_ADDR,
        G_ADDR,
        X_COORD,
        Y_COORD,
        VISIT_AREA_TYPE_CD,
        VISIT_CHC_REASON_CD,
        DGSTFN,
        REVISIT_INTENTION,
        RCMDTN_INTENTION)
        VALUES (
            '{PRE_VAI.TRAVEL_ID}',
            {int(PRE_VAI.VISIT_ORDER)},
            {int(PRE_VAI.VISIT_AREA_ID)},
            '{PRE_VAI.VISIT_AREA_NM}',
            {int(PRE_VAI.YMD[0])},
            {int(PRE_VAI.YMD[1])},
            {int(PRE_VAI.YMD[2])},
            '{PRE_VAI.ROAD_NM_ADDR}',
            '{PRE_VAI.LOTNO_ADDR}',
            '{PRE_VAI.SG[0]}',
            '{PRE_VAI.SG[1]}',
            {int(PRE_VAI.X_COORD)},
            {int(PRE_VAI.Y_COORD)},
            {int(PRE_VAI.VISIT_AREA_TYPE_CD)},
            {int(PRE_VAI.VISIT_CHC_REASON_CD)},
            {int(PRE_VAI.DGSTFN)},
            {int(PRE_VAI.REVISIT_INTENTION)},
            {int(PRE_VAI.RCMDTN_INTENTION)}
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
    create_table(db_connect)

    VAI = get_data()

    PRE_VAI = preprocessing(VAI)

    generate_data(db_connect, PRE_VAI)