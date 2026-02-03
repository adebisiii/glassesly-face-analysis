from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import mediapipe as mp

# =========================
# FastAPI APP (ZORUNLU)
# =========================
app = FastAPI(title="Face Analysis API")

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "face-analysis"}

@app.get("/health")
def health():
    return {"ok": True}

# =========================
# MediaPipe Face Mesh
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================
# Utils
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def to_px(lm, w, h):
    return [float(lm.x * w), float(lm.y * h)]

def compute_confidence(bgr, landmarks_px):
    """
    landmarks_px: dict -> {"LE":[x,y], ...}
    returns: int 0..100
    """
    h, w = bgr.shape[:2]

    xs = [p[0] for p in landmarks_px.values() if p]
    ys = [p[1] for p in landmarks_px.values() if p]
    if not xs or not ys:
        return 0

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    face_area_ratio = (bbox_w * bbox_h) / (w * h)

    # --- Face size score ---
    if face_area_ratio < 0.03:
        size_score = 0.2
    elif face_area_ratio < 0.08:
        size_score = 0.6
    elif face_area_ratio <= 0.45:
        size_score = 1.0
    else:
        size_score = 0.75

    # --- Margin score ---
    margin = 0.03
    margin_score = sum([
        x1 > w * margin,
        x2 < w * (1 - margin),
        y1 > h * margin,
        y2 < h * (1 - margin)
    ]) / 4.0

    # --- Blur score ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = 1.0 if lap_var > 80 else 0.6 if lap_var > 40 else 0.2

    # --- Exposure score ---
    mean = gray.mean()
    if mean < 50:
        expo_score = 0.35
    elif mean < 70:
        expo_score = 0.7
    elif mean <= 180:
        expo_score = 1.0
    elif mean <= 210:
        expo_score = 0.75
    else:
        expo_score = 0.45

    # --- Landmark sanity ---
    LE, RE = landmarks_px.get("LE"), landmarks_px.get("RE")
    if LE and RE:
        eye_dist = abs(RE[0] - LE[0])
        sanity_score = 1.0 if eye_dist > (bbox_w * 0.15) else 0.5
    else:
        sanity_score = 0.5

    score = (
        size_score   * 0.30 +
        margin_score * 0.20 +
        blur_score   * 0.25 +
        expo_score   * 0.15 +
        sanity_score * 0.10
    )

    return int(round(clamp(score, 0.0, 1.0) * 100))

# =========================
# API Endpoint
# =========================
@app.post("/v1/face-mesh")
async def face_mesh_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    img_arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return {"ok": False, "error": "invalid_image"}

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return {
            "ok": True,
            "confidence": 0,
            "landmarks": None
        }

    face = result.multi_face_landmarks[0].landmark

    # MediaPipe index referansları
    idx = {
        "LE": 33,
        "RE": 263,
        "NL": 97,
        "NR": 326,
        "FH": 10,
        "CT": 152,
        "FL": 234,
        "FR": 454
    }

    landmarks_px = {k: to_px(face[i], w, h) for k, i in idx.items()}

    confidence = compute_confidence(bgr, landmarks_px)

    return {
        "ok": True,
        "image": {"w": w, "h": h},
        "landmarks": landmarks_px,
        "confidence": confidence
    }
