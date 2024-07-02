FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install pandas psycopg2-binary python-dotenv

COPY visitareainfo_data_insertion.py visitareainfo_data_insertion.py
COPY .env .env
COPY Data/ Data/

ENTRYPOINT ["python", "visitareainfo_data_insertion.py"]