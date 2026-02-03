from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Any, Optional, List, Tuple

# =========================
# FastAPI APP
# =========================
app = FastAPI(title="Face Analysis API", version="2.0.0")

# CORS (Next.js için pratik)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod'da domain'e indir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Health check
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "face-analysis", "version": app.version}

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
def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def to_px(lm, w: int, h: int) -> List[float]:
    return [float(lm.x * w), float(lm.y * h)]

def dist(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if not a or not b:
        return None
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))

def bbox_from_points(points: Dict[str, Optional[List[float]]], w: int, h: int) -> Optional[Dict[str, float]]:
    xs = [p[0] for p in points.values() if p]
    ys = [p[1] for p in points.values() if p]
    if not xs or not ys:
        return None
    x1, x2 = float(min(xs)), float(max(xs))
    y1, y2 = float(min(ys)), float(max(ys))
    # clamp
    x1 = clamp(x1, 0.0, float(w-1))
    x2 = clamp(x2, 0.0, float(w-1))
    y1 = clamp(y1, 0.0, float(h-1))
    y2 = clamp(y2, 0.0, float(h-1))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2-x1, "h": y2-y1}

def compute_image_quality(bgr: np.ndarray, bbox: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """
    Basit ama stabil kalite metrikleri:
    - lap_var: blur (Laplacian variance)
    - mean_brightness: exposure
    - face_area_ratio
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())

    face_area_ratio = None
    if bbox and bbox["w"] > 0 and bbox["h"] > 0:
        face_area_ratio = float((bbox["w"] * bbox["h"]) / (w * h))

    return {
        "lap_var": lap_var,
        "mean_brightness": mean_brightness,
        "face_area_ratio": face_area_ratio
    }

def compute_confidence(
    bgr: np.ndarray,
    bbox: Optional[Dict[str, float]],
    landmarks_px: Dict[str, Optional[List[float]]]
) -> int:
    """
    returns: int 0..100
    """
    h, w = bgr.shape[:2]
    quality = compute_image_quality(bgr, bbox)
    lap_var = quality["lap_var"]
    mean = quality["mean_brightness"]
    face_area_ratio = quality["face_area_ratio"] or 0.0

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
    if bbox:
        margin = 0.03
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        margin_score = sum([
            x1 > w * margin,
            x2 < w * (1 - margin),
            y1 > h * margin,
            y2 < h * (1 - margin)
        ]) / 4.0
    else:
        margin_score = 0.4

    # --- Blur score ---
    blur_score = 1.0 if lap_var > 80 else 0.6 if lap_var > 40 else 0.2

    # --- Exposure score ---
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
    LE_IN, RE_IN = landmarks_px.get("LE_IN"), landmarks_px.get("RE_IN")
    if bbox and LE_IN and RE_IN:
        eye_d = dist(LE_IN, RE_IN) or 0.0
        bbox_w = max(1.0, bbox["w"])
        sanity_score = 1.0 if eye_d > (bbox_w * 0.12) else 0.5
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

def estimate_pose_flags(landmarks_px: Dict[str, Optional[List[float]]]) -> Dict[str, Any]:
    """
    Çok kaba flag’ler (yaw/roll için):
    - roll: gözlerin y farkı (normalize)
    - yaw: gözler arası orta nokta ile burun eksen kayması (normalize)
    """
    LE_IN = landmarks_px.get("LE_IN")
    RE_IN = landmarks_px.get("RE_IN")
    NOSE_TIP = landmarks_px.get("NOSE_TIP")

    flags = {"roll_score": None, "yaw_score": None}

    if LE_IN and RE_IN:
        eye_dist = dist(LE_IN, RE_IN) or 0.0
        if eye_dist > 1.0:
            roll = abs(RE_IN[1] - LE_IN[1]) / eye_dist
            flags["roll_score"] = float(clamp(roll, 0.0, 1.0))

            if NOSE_TIP:
                mid_x = (LE_IN[0] + RE_IN[0]) / 2.0
                yaw = abs(NOSE_TIP[0] - mid_x) / eye_dist
                flags["yaw_score"] = float(clamp(yaw, 0.0, 1.0))

    return flags

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
            "image": {"w": w, "h": h},
            "confidence": 0,
            "landmarks": None,
            "bbox": None,
            "quality": None,
            "pose": None
        }

    face = result.multi_face_landmarks[0].landmark

    # ✅ Daha sağlam landmark seti
    # - göz: inner corners + outer corners
    # - burun: nostril outer + nose tip
    # - yüz genişliği: temple (şakak)
    # - yükseklik: forehead + chin
    idx = {
        "LE_IN": 133,
        "RE_IN": 362,
        "LE_OUT": 33,
        "RE_OUT": 263,

        "NL": 94,
        "NR": 331,
        "NOSE_TIP": 1,

        "FH": 10,
        "CT": 152,

        "FL": 127,   # left temple
        "FR": 356,   # right temple
    }

    landmarks_px: Dict[str, Optional[List[float]]] = {}
    for k, i in idx.items():
        try:
            landmarks_px[k] = to_px(face[i], w, h)
        except Exception:
            landmarks_px[k] = None

    bbox = bbox_from_points(landmarks_px, w, h)
    confidence = compute_confidence(bgr, bbox, landmarks_px)
    quality = compute_image_quality(bgr, bbox)
    pose = estimate_pose_flags(landmarks_px)

    # Basit quality flags
    quality_flags = {
        "too_blurry": quality["lap_var"] is not None and quality["lap_var"] < 40,
        "too_dark": quality["mean_brightness"] is not None and quality["mean_brightness"] < 55,
        "too_bright": quality["mean_brightness"] is not None and quality["mean_brightness"] > 210,
        "face_too_small": quality["face_area_ratio"] is not None and quality["face_area_ratio"] < 0.03,
        "high_roll": pose["roll_score"] is not None and pose["roll_score"] > 0.18,
        "high_yaw": pose["yaw_score"] is not None and pose["yaw_score"] > 0.22,
    }

    return {
        "ok": True,
        "image": {"w": w, "h": h},
        "landmarks": landmarks_px,
        "bbox": bbox,
        "quality": quality,
        "pose": pose,
        "quality_flags": quality_flags,
        "confidence": confidence
    }
