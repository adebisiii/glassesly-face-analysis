import mediapipe as mp
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2


app = FastAPI()

# 2️⃣ Health & root endpoint'leri (BURAYA EKLENECEK)
@app.get("/")
def root():
    return {"ok": True, "service": "face-mesh"}

@app.get("/health")
def health():
    return {"ok": True}


mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,   # iris vb. daha iyi
    min_detection_confidence=0.5
)

def to_px(lm, w, h):
    return [float(lm.x * w), float(lm.y * h)]

@app.post("/v1/face-mesh")
async def face_mesh(file: UploadFile = File(...)):
    data = await file.read()
    img_arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"ok": False, "error": "invalid_image"}

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return {"ok": True, "confidence": 0, "landmarks": None}

    face = res.multi_face_landmarks[0].landmark  # 468 nokta

    # Senin schema'na yakın temel noktalar (yaklaşık indexler):
    # Sol göz dış/ iç, sağ göz dış/ iç vb. indexleri projene göre netleştirmen lazım.
    # Burada örnek olarak birkaç referans:
    idx = {
        "LE": 33,   # left eye outer corner
        "RE": 263,  # right eye outer corner
        "NL": 97,   # nose left (yaklaşık)
        "NR": 326,  # nose right (yaklaşık)
        "FH": 10,   # forehead/glabella üst
        "CT": 152,  # chin
        "FL": 234,  # left face edge (yaklaşık)
        "FR": 454   # right face edge (yaklaşık)
    }

    out = {k: to_px(face[i], w, h) for k, i in idx.items()}

    return {
        "ok": True,
        "image": {"w": w, "h": h},
        "landmarks": out,
        "confidence": 85  # bunu istersen kalite metriklerine göre düşür
    }
