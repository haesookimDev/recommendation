import psycopg2
from dotenv import load_dotenv
import os

HOST_URL = os.environ.get('HOST_URL')
USERNAME = os.environ.get('USERNAME')
USERPASSWORD = os.environ.get('USERPASSWORD')
DBNAME = os.environ.get('DBNAME')
PORT = os.environ.get('PORT')

db_connect = psycopg2.connect(
    user=USERNAME, 
    password=USERPASSWORD,
    host=HOST_URL,
    port=PORT,
    database=DBNAME,
)

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS traveller (
        id SERIAL PRIMARY KEY,
        TRAVELER_ID CHAR(7),
        GENDER CHAR(1),
        AGE_GRP NUMBER(3),
        TRAVEL_LIKE_SIDO_1 NUMBER(2),
        TRAVEL_LIKE_SGG_1 NUMBER(5),
        TRAVEL_LIKE_SIDO_2 NUMBER(2),
        TRAVEL_LIKE_SGG_2 NUMBER(5),
        TRAVEL_LIKE_SIDO_3 NUMBER(2),
        TRAVEL_LIKE_SGG_3 NUMBER(5),
        TRAVEL_STYL_1 NUMBER(1),
        TRAVEL_STYL_2 NUMBER(1),
        TRAVEL_STYL_3 NUMBER(1),
        TRAVEL_STYL_4 NUMBER(1),
        TRAVEL_STYL_5 NUMBER(1),
        TRAVEL_STYL_6 NUMBER(1),
        TRAVEL_STYL_7 NUMBER(1),
        TRAVEL_STYL_8 NUMBER(1),
        TRAVEL_STATUS_DESTINATION VARCHAR(4),
        TRAVEL_MOTIVE_1 NUMBER(10),
        TRAVEL_MOTIVE_2 NUMBER(10),
        TRAVEL_MOTIVE_3 NUMBER(10)
    );
    CREATE TABLE IF NOT EXISTS travel (
        id SERIAL PRIMARY KEY,
        TRAVELER_ID CHAR(7),
        TRAVEL_ID CHAR(9),
        AGE_GRP NUMBER(3),
        TRAVEL_PERIOD number(2)
    );
    CREATE TABLE IF NOT EXISTS travel_purpose (
        id SERIAL PRIMARY KEY,
        TRAVEL_ID CHAR(9),
        SHOPPING NUMBER(1),
        PARK NUMBER(1),
        HISTORY NUMBER(1),
        TOUR NUMBER(1),
        SPORTS NUMBER(1),
        ARTS NUMBER(1),
        PLAY NUMBER(1),
        CAMPING NUMBER(1),
        FESTIVAL NUMBER(1),
        SPA NUMBER(1),
        EDUCATION NUMBER(1),
        DRAMA NUMBER(1),
        PILGRIMAGE NUMBER(1),
        WELL NUMBER(1),
        SNS NUMBER(1),
        HOTEL NUMBER(1),
        NEWPLACE NUMBER(1),
        WITHPET NUMBER(1),
        MIMIC NUMBER(1),
        ECO NUMBER(1),
        HIKING NUMBER(1)
    );
    CREATE TABLE IF NOT EXISTS visit_area_info (
        id SERIAL PRIMARY KEY,
        TRAVEL_ID CHAR(9),
        VISIT_ORDER NUMBER(2),
        VISIT_AREA_ID NUMBER(10),
        VISIT_AREA_NM LONG,
        VISIT_START_Y NUMBER(4),
        VISIT_START_M NUMBER(2),
        VISIT_START_D NUMBER(2),
        ROAD_NM_ADDR LONG,
        LOTNO_ADDR LONG,
        X_COORD NUMBER(3,6),
        Y_COORD NUMBER(2,6),
        VISIT_AREA_TYPE_CD NUMBER(2),
        VISIT_CHC_REASON_CD NUMBER(2),
        DGSTFN NUMBER(1),
        REVISIT_INTENTION NUMBER(1),
        RCMDTN_INTENTION NUMBER(1),
    );
    """
    print("create_table_query")
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()