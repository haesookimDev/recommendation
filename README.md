# Recommendation System Project

## 프로젝트 개요
이 프로젝트는 여행 추천 시스템을 구축하는 것을 목표로 합니다. 여행자와 여행지 데이터를 기반으로 여행자에게 다음 여행지를 추천하는 시스템을 개발합니다.

## 환경 설정 및 요구사항
### 기본 요구사항
- Python 3.9
- Docker 및 Docker Compose

### 필수 패키지
- MLflow
- FastAPI
- PyTorch & PyTorch Geometric
- pandas
- scikit-learn
- psycopg2-binary
- boto3

## 주요 구성 요소

### 데이터베이스
- PostgreSQL을 사용하여 여행자, 여행, 방문지 정보를 저장합니다.
- 주요 테이블:
  - `traveller`: 여행자 기본 정보 (성별, 나이, 선호 지역, 여행 스타일)
  - `travel`: 여행 정보 (여행자 ID, 여행 ID, 여행 기간)
  - `travel_purpose`: 여행 목적 (쇼핑, 공원, 역사, 관광 등 22개 카테고리)
  - `visit_area_info`: 방문 장소 상세 정보 (좌표, 주소, 만족도, 재방문 의향)

### 전처리
- 여행 로그 데이터를 전처리하여 모델 학습에 적합한 형식으로 변환합니다.
- 데이터 전처리 과정:
  - 성별 인코딩 (남=1, 여=2)
  - 여행 스타일 및 동기 PCA 차원 축소
  - 주소 데이터에서 시/도, 군/구 정보 추출
  - 결측치 0으로 대체
  - 여행 기간 계산 및 정규화

### 추천 시스템
- 그래프 신경망(GNN)을 활용하여 여행자와 여행지 간의 관계를 학습합니다.
- 모델 아키텍처:
  - 6-layer Graph Convolutional Network (GCNConv)
  - Multi-head Attention 메커니즘
  - Dropout (p=0.5) 적용
  - L2 정규화
- 추천 방식:
  - Cosine Similarity 기반 여행지 유사도 계산
  - Contrastive Learning 손실 함수 사용
  - 여행자-여행 임베딩 결합을 통한 다음 여행지 예측

### API
- FastAPI를 사용하여 추천 시스템을 서비스로 제공합니다.
- 엔드포인트:
  - POST `/predict`: 여행지 추천 API
    - 입력: 여행자 정보 (성별, 나이 등), 여행 정보 (목적, 기간 등)
    - 출력: 추천 여행지 ID, 예상 평점, 추천 점수, 재방문 예상

### MLFlow
- 머신러닝 실험을 관리하고 추적하는 플랫폼입니다.
- 구성:
  - PostgreSQL 백엔드 스토어: 실험 메타데이터 저장
  - MinIO 아티팩트 스토어: 모델 가중치 및 결과물 저장 (S3 호환)
  - 실험 추적: 손실 함수 값, 모델 성능 지표 등 기록
  - 모델 버전 관리 및 배포

## 배포 가이드
### 환경 변수 설정
1. 각 컴포넌트별 `.env` 파일 구성:
   - Database: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `PORT`
   - MLFlow: `MLFLOW_USER`, `MLFLOW_PASSWORD`, `MLFLOW_DB`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### 서비스 실행
1. Docker Compose를 사용하여 서비스 실행:
   ```bash
   # 데이터베이스 서비스 시작
   cd DataBase && docker-compose up -d
   
   # MLFlow 서비스 시작
   cd MLFlow && docker-compose up -d
   
   # API 서비스 시작
   cd API && docker-compose up -d
   ```

2. 서비스 접근:
   - FastAPI 문서: `http://localhost:8000/docs`
   - MLFlow UI: `http://localhost:5001`
   - MinIO Console: `http://localhost:9001`

### 네트워크 구성
- 모든 서비스는 `mlops-network`라는 Docker 네트워크를 통해 통신
- 주요 포트:
  - FastAPI: 8000
  - MLFlow: 5001
  - MinIO: 9000(API), 9001(Console)
  - PostgreSQL: 설정된 PORT 값
