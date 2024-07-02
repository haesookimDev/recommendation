import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

HOST_URL = os.getenv('HOST_URL')
USERNAME = os.getenv('POSTGRES_USER')
USERPASSWORD = os.getenv('POSTGRES_PASSWORD')
DBNAME = os.getenv('POSTGRES_DB')
PORT = os.getenv('PORT')

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
        TRAVEL_STATUS_DESTINATION VARCHAR(4),
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

if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user=USERNAME, 
        password=USERPASSWORD,
        host=HOST_URL,
        port=PORT,
        database=DBNAME,
    )
    create_table(db_connect)