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
