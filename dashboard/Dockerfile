FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU 전용 PyTorch 먼저 설치 (GPU 버전 자동 설치 방지 — 이미지 크기 ~1.5GB 절감)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu  
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
# CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
