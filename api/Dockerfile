FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU 전용 PyTorch 먼저 설치 (GPU 버전 자동 설치 방지 — 이미지 크기 ~1.5GB 절감)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu  
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# FastAPI 서버 실행 명령어
# relode 옵션은 개발용으로 배포 시 제거해야됨.