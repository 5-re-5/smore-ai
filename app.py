import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from ultralytics import YOLO
import tensorflow as tf
from keras.layers import TFSMLayer
import mediapipe as mp
import numpy as np
import cv2
from collections import deque
import datetime
import jwt
from jwt import InvalidTokenError
import base64
from flask_cors import CORS

from sqlalchemy import create_engine, text

# ─── 환경 변수 로드 ───────────────────────────────────────────────────────────────
load_dotenv()
JWT_SECRET    = os.getenv("ACCESS_TOKEN_SECRET_KEY")
key_bytes     = base64.b64decode(JWT_SECRET.strip())
JWT_ALGORITHM = "HS256"

# ─── 에러 반환 정리 ───────────────────────────────────────────────────────────────
ERROR_400 = ({'error': '잘못된 요청'}, 400)
ERROR_401 = ({'error': '인증 실패(토큰 만료/없음)'}, 401)
ERROR_403 = ({'error': '권한 부족'}, 403)
ERROR_404 = ({'error': '사용자 없음'}, 404)
ERROR_500 = ({'error': '서버 내부 오류'}, 500)

# ─── SQLAlchemy 엔진(커넥션 풀) 설정 ───────────────────────────────────────────────
DB_USER = os.getenv("DEV_DB_USERNAME", "your_username")
DB_PASS = os.getenv("DEV_DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DEV_DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DEV_DB_PORT", "3306")
DB_NAME = os.getenv("DEV_DB_NAME", "your_db_name")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=20,         # 풀에 유지할 커넥션 수
    max_overflow=10,      # 풀 초과 시 추가 생성 가능한 수
    pool_timeout=30,      # 커넥션 얻기 대기 시간(초)
    pool_recycle=1800,    # 유휴 커넥션 재검증 주기(초)
    pool_pre_ping=True,   # 유휴 커넥션 검증
    connect_args={
        "ssl": {"fake_flag_to_enable_tls": True}
    }
)

# ─── Flask 앱 생성 ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='/')
CORS(app,
     supports_credentials=True,
     origins=[
       "http://127.0.0.1:5500",   # 기존 테스트 페이지
       "http://localhost:3000"    # React 등 개발 서버
     ])

# ─── 모델 경로 및 파라미터 ─────────────────────────────────────────────────────────
BASE = '/webcam-ai/runs'
MODEL_PATH   = os.path.join(BASE, 'yolo_engagement_m', 'weights', 'best.pt')
PHONE_MODEL  = os.path.join(BASE, 'mobile_phone_v8n', 'weights', 'best.pt')
TF_MODEL_DIR = os.path.join(BASE, 'posture_detection_model', 'model.savedmodel')

EM_WEIGHT      = 0.2
BL_WEIGHT      = 0.3
POSTURE_WEIGHT = 0.5
THRESHOLD      = 0.5
PHONE_PENALTY  = 0.6
OPEN_THRESH    = 0.25
CLOSED_THRESH  = 0.20
MIN_SAMPLES    = 5
MAX_IMAGE_SIZE = 100 * 1024       # 100KB
ALLOWED_MIMES  = {'image/jpeg', 'image/png'}

# ─── 모델 로드 ─────────────────────────────────────────────────────────────────
yolo        = YOLO(MODEL_PATH)
phone_model = YOLO(PHONE_MODEL)

# SavedModel 을 TFSMLayer 로 래핑
posture_layer = TFSMLayer(
    TF_MODEL_DIR,
    call_endpoint='serving_default'
)
posture_model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    posture_layer
])

# ─── MediaPipe FaceMesh, EAR 계산 설정 ───────────────────────────────────────────
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)
LEFT_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_IDX = [362, 385, 387, 263, 373, 380]
user_ears = {}  # { user_id: deque([...], maxlen=MIN_SAMPLES) }

def compute_ear(landmarks, w, h):
    def _ear(idxs):
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idxs]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (A + B) / (2.0 * C)
    return (_ear(LEFT_IDX) + _ear(RIGHT_IDX)) / 2.0

@app.route('/ai/focus-records', methods=['POST'])
def predict_meta():
    try:
        # 0) Content-Type 검사
        if 'multipart/form-data' not in (request.content_type or ''):
            return ERROR_400

        # 0.1) 토큰 추출·검증
        token = request.cookies.get('accessToken')
        if not token:
            return ERROR_401
        try:
            payload = jwt.decode(token, key_bytes, algorithms=[JWT_ALGORITHM])
        except InvalidTokenError as e:
            app.logger.error(f"JWT decode error: {e}")
            return ERROR_401
        user_id_token = int(payload.get('sub', -1))

        # 1) user_id 검증
        user_id_form = int(request.form.get('user_id', -1))
        if user_id_form != user_id_token or user_id_form <= 0:
            return ERROR_403 if user_id_form > 0 else ERROR_400

        # DB에 유저 존재 확인
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM users WHERE user_id = :uid"),
                {"uid": user_id_form}
            )
            if result.scalar() is None:
                return ERROR_404

        # 2) 이미지 검증
        if 'image' not in request.files:
            return ERROR_400
        file = request.files['image']
        if file.mimetype not in ALLOWED_MIMES:
            return ERROR_400
        stream = file.stream
        stream.seek(0, os.SEEK_END)
        if stream.tell() > MAX_IMAGE_SIZE:
            return ERROR_400
        stream.seek(0)

        # 3) 이미지 로드
        buf = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        # 4a) 휴대폰 탐지
        phone_res      = phone_model.predict(source=img, imgsz=224)[0]
        cls_ids        = phone_res.boxes.cls.cpu().numpy().astype(int)
        phone_detected = 0 in cls_ids

        # 4b) 감정(YOLO)
        y_res = yolo.predict(source=img, imgsz=224)[0]
        if len(y_res.boxes) and 0 in y_res.boxes.cls.cpu().numpy().astype(int):
            confs = y_res.boxes.conf.cpu().numpy()
            cls   = y_res.boxes.cls.cpu().numpy().astype(int)
            p_em  = float(np.mean(confs[cls == 0]))
        else:
            p_em = 0.0

        # 4c) 자세(TF SavedModel via TFSMLayer)
        rgb224  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb224, (224,224)).astype(np.float32) / 255.0
        batch   = np.expand_dims(resized, axis=0)
        preds   = posture_model.predict(batch, verbose=0)
        app.logger.debug(f"posture preds: {type(preds)} {preds}")
        if isinstance(preds, dict):
            key       = list(preds.keys())[0]
            array_out = preds[key]
            p_posture = float(array_out[0][0])
        else:
            arr       = preds.numpy() if hasattr(preds, 'numpy') else preds
            p_posture = float(arr[0][0])

        # 4d) 깜빡임(EAR)
        det = face_mesh.process(rgb224)
        if not det.multi_face_landmarks:
            p_bl = 0.0
        else:
            ear       = compute_ear(det.multi_face_landmarks[0].landmark, w, h)
            buf_deque = user_ears.setdefault(str(user_id_token), deque(maxlen=MIN_SAMPLES))
            buf_deque.append(ear)
            ears      = list(buf_deque)
            if len(ears) < MIN_SAMPLES:
                p_bl = 0.0
            else:
                blinks    = sum(
                    1 for i in range(1, MIN_SAMPLES)
                    if ears[i-1] > OPEN_THRESH and ears[i] < CLOSED_THRESH
                )
                focus_pct = 0 if all(e < CLOSED_THRESH for e in ears) else round(max(0, min(100, (1 - blinks/2)*100)), 1)
                p_bl      = focus_pct / 100.0

        # 5) 앙상블 집중도
        ensemble_score = EM_WEIGHT * p_em + BL_WEIGHT * p_bl + POSTURE_WEIGHT * p_posture
        if phone_detected:
            ensemble_score *= (1 - PHONE_PENALTY)
        status = round(ensemble_score * 100, 1)

        # 6) timestamp
        ts_str = request.form.get('timestamp')
        if ts_str:
            ts = datetime.datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ')
        else:
            ts = datetime.datetime.now(datetime.timezone.utc)

        # 7) DB 저장
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO focus_records (user_id, timestamp, status)
                    VALUES (:uid, :ts, :status)
                """),
                {"uid": user_id_token, "ts": ts, "status": status}
            )
            record_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()

        # 8) JSON 응답
        return jsonify({
            "data": {
                "record": {
                    "record_id": record_id,
                    "user_id":   user_id_token,
                    "timestamp": ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "status":    status
                }
            }
        }), 200

    except Exception as e:
        app.logger.exception(f"predict_meta 처리 중 예외: {e}")
        return ERROR_500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)
