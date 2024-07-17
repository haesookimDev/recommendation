FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install mlflow pandas scikit-learn "fastapi[all]" torch_geometric

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY api.py api.py
COPY .env .env
COPY Recommender/ Recommender/
COPY method/ method/

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--reload"]