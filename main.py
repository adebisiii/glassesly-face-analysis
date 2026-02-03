def clamp(x, a, b):
    return max(a, min(b, x))

def compute_confidence(bgr, landmarks_px):
    """
    landmarks_px: dict -> {"LE":[x,y], ...} pixel coordinate
    returns: int 0..100
    """
    h, w = bgr.shape[:2]

    # --- 1) Face bbox from selected landmarks (approx) ---
    xs = [p[0] for p in landmarks_px.values() if p]
    ys = [p[1] for p in landmarks_px.values() if p]
    if not xs or not ys:
        return 0

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    bbox_area = bbox_w * bbox_h
    img_area = w * h
    face_area_ratio = bbox_area / img_area  # 0..1

    # Face size score: ideal aralık ~ %8 - %45 (projeye göre ayarlanır)
    # Çok küçükse düşür, çok büyükse de (çok yakın) biraz düşür.
    if face_area_ratio < 0.03:
        size_score = 0.2
    elif face_area_ratio < 0.08:
        size_score = 0.6
    elif face_area_ratio <= 0.45:
        size_score = 1.0
    else:
        size_score = 0.75

    # --- 2) Margin score (yüz kenarlara çok yakın mı?) ---
    margin = 0.03  # %3
    left_ok   = x1 > w * margin
    right_ok  = x2 < w * (1 - margin)
    top_ok    = y1 > h * margin
    bottom_ok = y2 < h * (1 - margin)

    # 4 kenardan kaç tanesi ok
    ok_count = sum([left_ok, right_ok, top_ok, bottom_ok])
    margin_score = ok_count / 4.0  # 0..1

    # --- 3) Blur score (variance of Laplacian) ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Tipik: 50 altı bulanık, 120+ iyi (cihaz değişir)
    if lap_var < 40:
        blur_score = 0.2
    elif lap_var < 80:
        blur_score = 0.6
    else:
        blur_score = 1.0

    # --- 4) Exposure score (brightness) ---
    mean = float(gray.mean())
    # ideal ~ 70-180 arası; çok karanlık veya çok aydınlık düşür
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

    # --- 5) Landmark sanity score (gözler çok yakın vs.) ---
    # basit kontrol: göz arası mesafe > bbox_w'nin %15'i olmalı
    LE, RE = landmarks_px.get("LE"), landmarks_px.get("RE")
    if LE and RE:
        eye_dist = abs(RE[0] - LE[0])
        sanity_score = 1.0 if eye_dist > (bbox_w * 0.15) else 0.5
    else:
        sanity_score = 0.5

    # --- Weighted combine ---
    score = (
        size_score   * 0.30 +
        margin_score * 0.20 +
        blur_score   * 0.25 +
        expo_score   * 0.15 +
        sanity_score * 0.10
    )

    confidence = int(round(clamp(score, 0.0, 1.0) * 100))
    return confidence
