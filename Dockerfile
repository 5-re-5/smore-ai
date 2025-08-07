# 1) 베이스 이미지
FROM python:3.10-slim

# 2) 작업 디렉터리 (이 아래에 app.py, requirements.txt가 있다고 가정)
WORKDIR /smore-ai

# 3) 시스템 의존성 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 4) 파이썬 패키지 복사 및 설치
COPY requirements.txt ./  
RUN pip install --no-cache-dir -r requirements.txt

# 5) 애플리케이션 코드 및 모델 복사
COPY . .  

# 6) Flask 환경 변수
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=8082

# 7) 포트 오픈
EXPOSE 8082

# 8) 컨테이너 구동 명령
CMD ["flask", "run"]