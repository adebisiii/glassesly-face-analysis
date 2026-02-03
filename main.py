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

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Face Mesh UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 20px; max-width: 1100px; margin: 0 auto; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-top: 16px; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #ccc; background: #f7f7f7; cursor: pointer; }
    pre { background: #0b1020; color: #d7e1ff; padding: 12px; border-radius: 10px; overflow: auto; max-height: 420px; }
    .row { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
    .hint { color: #666; font-size: 14px; }
    canvas { width: 100%; height: auto; border-radius: 12px; border: 1px solid #eee; background: #fafafa; }
    .pill { display:inline-block; padding: 2px 8px; border:1px solid #ddd; border-radius:999px; font-size: 12px; margin-left: 8px; color:#333; }
  </style>
</head>
<body>
  <h1>Face Mesh UI <span class="pill">canvas overlay</span></h1>
  <p class="hint">Fotoğraf seç → Analyze → noktalar resmin üstüne çizilir. Sağda JSON’u görürsün.</p>

  <div class="card">
    <input id="file" type="file" accept="image/*"/>
    <button id="run">Analyze</button>
    <span id="status" class="pill"></span>
  </div>

  <div class="row">
    <div class="card">
      <h3>Preview + Landmarks</h3>
      <canvas id="cv"></canvas>
      <p class="hint">İpucu: LE/RE göz köşelerinde, NL/NR burun kanatlarında, FL/FR yüz kenarında, FH alın üstünde, CT çenede olmalı.</p>
    </div>
    <div class="card">
      <h3>Result JSON</h3>
      <pre id="out">{}</pre>
    </div>
  </div>

<script>
const fileEl = document.getElementById('file');
const runBtn = document.getElementById('run');
const outEl = document.getElementById('out');
const statusEl = document.getElementById('status');
const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');

let img = new Image();
let imgURL = null;

function setStatus(t){ statusEl.textContent = t || ""; }

function drawImageToCanvas() {
  if (!img || !img.naturalWidth) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0);
}

function drawPoint(x, y, label) {
  const r = Math.max(4, Math.round(canvas.width * 0.006)); // ölçeğe göre
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 160, 255, 0.95)";
  ctx.fill();

  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
  ctx.stroke();

  // label
  ctx.font = `${Math.max(14, Math.round(canvas.width * 0.02))}px system-ui`;
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.fillText(label, x + r + 4, y - r - 4);
}

function drawAllLandmarks(landmarks) {
  if (!landmarks) return;
  const order = ["LE","RE","NL","NR","FL","FR","FH","CT"];
  for (const k of order) {
    const p = landmarks[k];
    if (Array.isArray(p) && p.length === 2) {
      drawPoint(Number(p[0]), Number(p[1]), k);
    }
  }
  // yardımcı: gözler arası çizgi
  if (landmarks.LE && landmarks.RE) {
    ctx.beginPath();
    ctx.moveTo(landmarks.LE[0], landmarks.LE[1]);
    ctx.lineTo(landmarks.RE[0], landmarks.RE[1]);
    ctx.lineWidth = Math.max(2, Math.round(canvas.width * 0.003));
    ctx.strokeStyle = "rgba(255, 80, 80, 0.85)";
    ctx.stroke();
  }
  // yardımcı: yüz genişliği çizgisi
  if (landmarks.FL && landmarks.FR) {
    ctx.beginPath();
    ctx.moveTo(landmarks.FL[0], landmarks.FL[1]);
    ctx.lineTo(landmarks.FR[0], landmarks.FR[1]);
    ctx.lineWidth = Math.max(2, Math.round(canvas.width * 0.003));
    ctx.strokeStyle = "rgba(80, 200, 120, 0.85)";
    ctx.stroke();
  }
}

fileEl.addEventListener('change', () => {
  const f = fileEl.files?.[0];
  if (!f) return;

  if (imgURL) URL.revokeObjectURL(imgURL);
  imgURL = URL.createObjectURL(f);
  img = new Image();
  img.onload = () => drawImageToCanvas();
  img.src = imgURL;

  outEl.textContent = "{}";
  setStatus("ready");
});

runBtn.addEventListener('click', async () => {
  const f = fileEl.files?.[0];
  if (!f) { alert("Lütfen bir fotoğraf seç"); return; }

  setStatus("analyzing...");
  outEl.textContent = "{}";

  // önce resmi çiz
  drawImageToCanvas();

  const fd = new FormData();
  fd.append("file", f);

  try {
    const res = await fetch("/v1/face-mesh", { method: "POST", body: fd });
    const json = await res.json();
    outEl.textContent = JSON.stringify(json, null, 2);

    // overlay çiz
    drawImageToCanvas();
    if (json && json.landmarks) drawAllLandmarks(json.landmarks);

    setStatus("done");
  } catch (e) {
    setStatus("error");
    outEl.textContent = String(e);
  }
});
</script>
</body>
</html>
"""


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
