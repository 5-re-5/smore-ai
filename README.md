# Webcam-AI Focus Detection API

자체 개발한 YOLO 기반 감정·휴대폰·자세·눈 깜빡임 앙상블 모델을 이용해  
업로드된 이미지를 분석하고, 사용자 집중도를 기록하는 Flask API 서버입니다.
### 집중도 점수?
평소 집중 상태를 100점으로 둔 뒤,
집중하지 않는 상태들을 점검하고 점수를 깎는 방식으로 점수 계산
예) 휴대폰 사용감지 -> 전체 점수에서 60% 차감

---

## 🔖 주요 기능

- **감정(Emotion) 탐지**: YOLO 모델(`yolo_engagement_m`)  
- **휴대폰 사용 탐지**: YOLO 모델(`mobile_phone_v8n`)  
- **자세(Posture) 분석**: TensorFlow SavedModel + TFSMLayer  
- **눈 깜빡임(Blink) 계산**: MediaPipe FaceMesh로 EAR(Eye Aspect Ratio) 산출  
- **앙상블 집중도 계산**: 가중합 + 휴대폰 페널티 적용  
- **결과 DB 저장**: MySQL (SQLAlchemy 커넥션 풀)  

---

## 📦 요구 사항

- Python 3.8 이상
- MySQL 또는 호환 가능한 데이터베이스
- GPU가 탑재된 환경(선택 사항)  

### 주요 파이썬 패키지

- `flask`
- `flask-cors`
- `python-dotenv`
- `ultralytics`
- `tensorflow`
- `keras`
- `mediapipe`
- `numpy`
- `opencv-python`
- `sqlalchemy`
- `pymysql`
- `PyJWT`

---

## ⚙️ 설치 및 실행

```bash
# 1) 저장소 클론
git clone https://github.com/your-org/webcam-ai-focus-api.git
cd webcam-ai-focus-api

# 2) 가상환경 생성 & 활성화 (선택)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3) 의존성 설치
pip install -r requirements.txt

# 4) 환경 변수 설정
cp .env.example .env
# .env 파일에 아래 변수들 채워주세요:
#
# ACCESS_TOKEN_SECRET_KEY=...
# DEV_DB_USERNAME=...
# DEV_DB_PASSWORD=...
# DEV_DB_HOST=...
# DEV_DB_PORT=3306
# DEV_DB_NAME=...

# 5) 모델 파일 위치 확인
# BASE 폴더 내에 다음 경로 구조로 모델이 있어야 합니다:
# /webcam-ai/runs/
#   ├─ yolo_engagement_m/weights/best.pt
#   ├─ mobile_phone_v8n/weights/best.pt
#   └─ posture_detection_model/model.savedmodel/...

# 6) 서버 실행
python app.py
# → http://0.0.0.0:8082 에서 API 대기
