"""
app.py  —  DeepFake Detection System  (Seminar / Viva Edition)
==============================================================
Pipeline:
    Upload  →  Haar Cascade face detection
            →  25 deterministic face variants
            →  EfficientNetB2 feature extraction  (1408-dim)
            →  LSTM temporal classification
            →  Calibrated threshold  →  REAL / FAKE verdict

Run:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Typography ────────────────────────────────────────────────────────── */
.title-text { font-size:2.2rem; font-weight:800; letter-spacing:-0.5px;
              margin-bottom:2px; }
.subtitle   { font-size:1rem; color:#8b949e; margin-top:0; margin-bottom:18px; }

/* ── Verdict cards ──────────────────────────────────────────────────────── */
.fake-card { background:linear-gradient(135deg,#7f1d1d,#ef4444);
             padding:28px 24px; border-radius:14px; color:#fff;
             text-align:center;
             box-shadow:0 4px 20px rgba(239,68,68,0.40); }
.real-card { background:linear-gradient(135deg,#14532d,#22c55e);
             padding:28px 24px; border-radius:14px; color:#fff;
             text-align:center;
             box-shadow:0 4px 20px rgba(34,197,94,0.40); }
.warn-card { background:linear-gradient(135deg,#78350f,#f59e0b);
             padding:28px 24px; border-radius:14px; color:#fff;
             text-align:center;
             box-shadow:0 4px 20px rgba(245,158,11,0.35); }

/* ── Simple progress bar ───────────────────────────────────────────────── */
.simple-bar-container {
    background-color: #e5e7eb;
    border-radius: 12px;
    height: 32px;
    overflow: hidden;
    margin: 10px 0;
}
.simple-bar-fake {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 12px;
    color: white;
    font-weight: bold;
    font-size: 14px;
    transition: width 0.3s ease;
}
.simple-bar-real {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-left: 12px;
    color: white;
    font-weight: bold;
    font-size: 14px;
    transition: width 0.3s ease;
}

/* ── Debug block ────────────────────────────────────────────────────────── */
.dbg { background:#0d1117; color:#c9d1d9; padding:14px; border-radius:8px;
       font-family:monospace; font-size:12px; white-space:pre;
       border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
SEQ_LEN      = 25
FEAT_DIM     = 1408
IMG_SIZE     = 224
MODEL_PATH   = "latest_EfficientNetB2.h5"
OPT_THR_FILE = os.path.join("eval_results", "optimal_threshold.txt")

_CROP_RATIOS  = [0.00, 0.04, 0.07, 0.09, 0.11]
_BRIGHTNESSES = [-20, -10, 0, 10, 20]
_LOCAL_WINDOW = 7


# ═════════════════════════════════════════════════════════════════════════════
#  THRESHOLD LOADER
# ═════════════════════════════════════════════════════════════════════════════
def _load_optimal_threshold():
    try:
        raw = float(open(OPT_THR_FILE).read().strip())
        if not np.isfinite(raw) or not (0.0 <= raw <= 1.0):
            return None
        return float(np.clip(raw, 0.20, 0.80))
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        from tensorflow.keras.applications import EfficientNetB2
        from tensorflow.keras.models import load_model
        clf      = load_model(MODEL_PATH)
        backbone = EfficientNetB2(weights="imagenet",
                                  include_top=False,
                                  pooling="avg")
        return clf, backbone, None
    except Exception as exc:
        return None, None, str(exc)


# ═════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING HELPER
# ═════════════════════════════════════════════════════════════════════════════
def _preprocess():
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input


# ═════════════════════════════════════════════════════════════════════════════
#  FACE DETECTION
# ═════════════════════════════════════════════════════════════════════════════
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face(frame: np.ndarray):
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = []
    for scale, min_n in [(1.05, 6), (1.10, 4), (1.20, 3)]:
        faces = _CASCADE.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=min_n,
            minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            break
    if not len(faces):
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mg   = int(0.10 * max(w, h))
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x - mg), max(0, y - mg)
    x2, y2 = min(fw, x + w + mg), min(fh, y + h + mg)
    face   = cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE),
                        interpolation=cv2.INTER_AREA)
    return face, (x, y, w, h)


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def extract_features(face_bgr: np.ndarray, backbone) -> np.ndarray:
    if face_bgr is None:
        return None
    pp       = _preprocess()
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_f32 = face_rgb.astype(np.float32)
    face_pp  = pp(face_f32)
    batch    = np.expand_dims(face_pp, axis=0)
    feat     = backbone.predict(batch, verbose=0)
    return feat[0].astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC FACE VARIANTS
# ═════════════════════════════════════════════════════════════════════════════
def make_face_variants(face_bgr: np.ndarray) -> list:
    variants = []
    h, w = face_bgr.shape[:2]
    for cr in _CROP_RATIOS:
        for bs in _BRIGHTNESSES:
            cx, cy = int(w * cr / 2), int(h * cr / 2)
            if cx > 0 or cy > 0:
                y2 = h - cy if cy > 0 else h
                x2 = w - cx if cx > 0 else w
                v  = cv2.resize(face_bgr[cy:y2, cx:x2],
                                (IMG_SIZE, IMG_SIZE),
                                interpolation=cv2.INTER_AREA)
            else:
                v = face_bgr.copy()
            v = np.clip(v.astype(np.int16) + bs, 0, 255).astype(np.uint8)
            variants.append(v)
    return variants


# ═════════════════════════════════════════════════════════════════════════════
#  MISSING-FEATURE INTERPOLATION
# ═════════════════════════════════════════════════════════════════════════════
def fill_missing(feat_list: list) -> np.ndarray:
    n   = len(feat_list)
    out = np.zeros((n, FEAT_DIM), dtype=np.float32)
    valid = [i for i, f in enumerate(feat_list) if f is not None]
    if not valid:
        return out
    for i in range(valid[0]):
        out[i] = feat_list[valid[0]]
    for k in range(len(valid)):
        ia = valid[k]
        out[ia] = feat_list[ia]
        if k + 1 < len(valid):
            ib = valid[k + 1]
            fa, fb = feat_list[ia], feat_list[ib]
            for j in range(ia + 1, ib):
                t = (j - ia) / (ib - ia)
                out[j] = (1.0 - t) * fa + t * fb
    for i in range(valid[-1] + 1, n):
        out[i] = feat_list[valid[-1]]
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
def run_model(clf, sequence: np.ndarray) -> float:
    seq = sequence.reshape(1, SEQ_LEN, FEAT_DIM).astype(np.float32)
    if len(clf.inputs) == 2:
        aux = np.zeros((1, int(clf.inputs[1].shape[1])), np.float32)
        out = clf.predict([seq, aux], verbose=0)
    else:
        out = clf.predict(seq, verbose=0)
    out = np.squeeze(out)
    if out.ndim == 0:
        return float(out)
    if out.ndim == 1 and out.shape[0] == 2:
        return float(out[1])
    return float(out[0])


# ═════════════════════════════════════════════════════════════════════════════
#  PER-FRAME SLIDING-WINDOW CONTEXT
# ═════════════════════════════════════════════════════════════════════════════
def per_frame_window(sequence: np.ndarray, i: int,
                     local_w: int = _LOCAL_WINDOW) -> np.ndarray:
    n    = len(sequence)
    half = local_w // 2
    s    = max(0, min(i - half, n - local_w))
    e    = min(n, s + local_w)
    if e - s < local_w:
        s = max(0, e - local_w)
    ctx = list(sequence[s:e])
    while len(ctx) < SEQ_LEN:
        ctx = ([ctx[0]] + ctx) if len(ctx) % 2 == 0 else (ctx + [ctx[-1]])
    return np.array(ctx[:SEQ_LEN], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  INTERPRET SCORE
# ═════════════════════════════════════════════════════════════════════════════
def _interpret(raw: float, thr: float):
    if raw >= thr:
        label, conf = "FAKE", float(raw)
    else:
        label, conf = "REAL", float(1.0 - raw)

    if raw < 0.30:
        band = "Strongly REAL"
    elif raw < 0.60:
        band = "Uncertain"
    else:
        band = "Strongly FAKE"

    return label, conf, band


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def analyze_image(clf, backbone, image_bgr: np.ndarray, threshold: float):
    face, _ = detect_face(image_bgr)

    if face is None:
        return ("UNCERTAIN", 0.0, 0.5, "No face detected",
                None, {"error": "No face detected"})

    variants  = make_face_variants(face)
    feat_list = [extract_features(v, backbone) for v in variants]
    sequence  = fill_missing(feat_list)
    raw       = run_model(clf, sequence)

    seq_var   = float(sequence.var())
    zero_rows = int(np.sum(sequence.sum(axis=1) == 0))
    const_out = seq_var < 1e-6

    dbg = {
        "raw_score": raw,
        "seq_mean": float(sequence.mean()),
        "seq_var": seq_var,
        "zero_rows": zero_rows,
        "constant_seq": const_out,
        "face_detected": True,
    }

    label, conf, band = _interpret(raw, threshold)
    return label, conf, raw, band, face, dbg


# ═════════════════════════════════════════════════════════════════════════════
#  VIDEO FRAME EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def extract_frames(path: str, n: int = SEQ_LEN):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0:
        idxs   = np.linspace(0, total - 1, n, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frm = cap.read()
            if ok:
                frames.append(frm)
        cap.release()
    else:
        raw = []
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            raw.append(frm)
        cap.release()
        total = len(raw)
        if not raw:
            return [], 0
        idxs   = np.linspace(0, total - 1, n, dtype=int)
        frames = [raw[i] for i in idxs]

    while frames and len(frames) < n:
        frames.append(frames[-1])

    return frames[:n], total


# ═════════════════════════════════════════════════════════════════════════════
#  VIDEO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def analyze_video(clf, backbone, path: str, threshold: float,
                  progress_ph=None):
    frames, total = extract_frames(path, SEQ_LEN)

    if not frames:
        return ("UNCERTAIN", 0.0, 0.5, "No frames", [], [], [],
                {"error": "No frames readable from video"})

    feat_list  = []
    face_boxes = []
    n_detected = 0

    for idx, frame in enumerate(frames):
        face, bbox = detect_face(frame)
        feat       = extract_features(face, backbone)
        feat_list.append(feat)
        face_boxes.append(bbox)
        if feat is not None:
            n_detected += 1
        if progress_ph:
            progress_ph.progress(
                (idx + 1) / SEQ_LEN,
                text=f"Extracting frame features… {idx+1}/{SEQ_LEN}")

    sequence    = fill_missing(feat_list)
    overall_raw = run_model(clf, sequence)

    frame_scores = [
        run_model(clf, per_frame_window(sequence, i))
        for i in range(SEQ_LEN)
    ]

    seq_var   = float(sequence.var())
    zero_rows = int(np.sum(sequence.sum(axis=1) == 0))

    dbg = {
        "raw_score": overall_raw,
        "total_frames": total,
        "sampled_frames": SEQ_LEN,
        "faces_detected": n_detected,
        "detection_rate": round(n_detected / SEQ_LEN, 3),
        "seq_mean": float(sequence.mean()),
        "seq_var": seq_var,
        "zero_rows": zero_rows,
    }

    label, conf, band = _interpret(overall_raw, threshold)
    return label, conf, overall_raw, band, frame_scores, frames, face_boxes, dbg


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotate_frame(frame, bbox, score, threshold):
    if bbox is None:
        return frame
    out = frame.copy()
    x, y, w, h = bbox
    col = (255, 60, 60) if score >= threshold else (60, 200, 60)
    cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
    tag = f"{'FAKE' if score >= threshold else 'REAL'}: {score:.1%}"
    cv2.putText(out, tag, (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  SIMPLIFIED VERDICT DISPLAY WITH EASY-TO-UNDERSTAND BAR
# ═════════════════════════════════════════════════════════════════════════════
def show_verdict(label: str, conf: float, raw: float,
                 band: str, threshold: float):
    # Verdict card
    if label == "FAKE":
        st.markdown(
            f'<div class="fake-card">'
            f'<h2 style="margin:0">🚨 DEEPFAKE DETECTED</h2>'
            f'<p style="margin:8px 0 0 0;opacity:0.85;font-size:1rem">{band}</p>'
            f'</div>', unsafe_allow_html=True)
    elif label == "REAL":
        st.markdown(
            f'<div class="real-card">'
            f'<h2 style="margin:0">✅ AUTHENTIC MEDIA</h2>'
            f'<p style="margin:8px 0 0 0;opacity:0.85;font-size:1rem">{band}</p>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warn-card">'
            '<h2 style="margin:0">⚠️ UNCERTAIN — No face found</h2>'
            '<p style="margin:8px 0 0 0;opacity:0.85">Upload a clear, '
            'front-facing image or video.</p>'
            '</div>', unsafe_allow_html=True)
        return

    st.markdown("")

    # Simple metric row (3 columns instead of 4 for clarity)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Verdict", label)
    with col2:
        st.metric("Confidence", f"{conf:.1%}")
    with col3:
        st.metric("Raw Score", f"{raw:.3f}")

    st.markdown("---")

    # SIMPLE, CLEAR PROGRESS BAR
    st.markdown(f"**Fake Probability Score: {raw:.1%}**")
    st.caption(f"(Threshold: {threshold:.2f} | Score ≥ {threshold:.2f} = FAKE)")
    
    fake_pct = int(raw * 100)
    real_pct = 100 - fake_pct
    
    if label == "FAKE":
        st.markdown(f"""
<div class="simple-bar-container">
    <div class="simple-bar-fake" style="width:{fake_pct}%">
        {fake_pct}% FAKE
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="simple-bar-container">
    <div class="simple-bar-real" style="width:{real_pct}%">
        {real_pct}% REAL
    </div>
</div>
""", unsafe_allow_html=True)

    # Visual threshold indicator
    st.markdown("---")
    st.markdown("**How to interpret:**")
    
    if raw >= threshold:
        st.markdown(f"- 🔴 Score ({raw:.3f}) is **ABOVE** threshold ({threshold:.2f}) → **FAKE**")
    else:
        st.markdown(f"- 🟢 Score ({raw:.3f}) is **BELOW** threshold ({threshold:.2f}) → **REAL**")

    # Brief commentary
    if label == "FAKE":
        if conf > 0.8:
            st.info("🎯 High confidence: Strong manipulation indicators detected")
        elif conf > 0.6:
            st.warning("⚠️ Medium confidence: Suspicious patterns found")
        else:
            st.warning("❓ Low confidence: Borderline case - consider manual review")
    else:
        if conf > 0.8:
            st.success("✅ High confidence: Media appears authentic")
        elif conf > 0.6:
            st.success("👍 Medium confidence: Likely authentic")
        else:
            st.warning("❓ Low confidence: Borderline case - consider manual review")


# ═════════════════════════════════════════════════════════════════════════════
#  FRAME SCORE CHART
# ═════════════════════════════════════════════════════════════════════════════
def plot_frame_scores(scores: list, threshold: float):
    colors = ["#ef4444" if s >= threshold else "#22c55e" for s in scores]
    fig    = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))), y=scores,
        mode="lines+markers", name="Fake probability",
        line=dict(color="#6699ff", width=2),
        marker=dict(color=colors, size=9,
                    line=dict(width=1, color="rgba(255,255,255,0.6)")),
        hovertemplate="Frame %{x}<br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#aaa",
                  annotation_text=f"Threshold ({threshold:.2f})",
                  annotation_position="bottom right")
    fig.update_layout(
        xaxis_title="Frame index",
        yaxis_title="Fake probability",
        yaxis=dict(range=[0, 1]),
        height=340,
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DEBUG PANEL
# ═════════════════════════════════════════════════════════════════════════════
def render_debug(dbg: dict):
    with st.expander("🔧 Debug panel", expanded=False):
        lines = "\n".join(
            f"  {k:<26}: {v:.6f}" if isinstance(v, float)
            else f"  {k:<26}: {v}"
            for k, v in dbg.items()
        )
        st.markdown(f'<div class="dbg">{lines}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL INSIGHTS PANEL
# ═════════════════════════════════════════════════════════════════════════════
def render_insights(dbg: dict):
    with st.expander("🔬 Model Insights", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Feature mean", f"{dbg.get('seq_mean', 0):.4f}")
        c2.metric("Feature variance", f"{dbg.get('seq_var', 0):.4f}")
        c3.metric("Zero-feature rows", f"{dbg.get('zero_rows', 0)}/{SEQ_LEN}")

        if dbg.get("seq_var", 1) < 1e-6:
            st.error(
                "⚠️ Feature variance is 0 — all 25 input variants produced "
                "identical embeddings. This usually means the LSTM model has "
                "untrained weights.")


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION RESULTS TAB
# ═════════════════════════════════════════════════════════════════════════════
def render_eval_tab():
    st.subheader("📊 Evaluation Results")
    st.caption("Run `python evaluate.py --dataset .` to populate this tab.")

    report = os.path.join("eval_results", "threshold_report.txt")
    cm_img = os.path.join("eval_results", "confusion_matrix.png")
    roc_img = os.path.join("eval_results", "roc_curve.png")
    dist_img = os.path.join("eval_results", "score_distribution.png")

    if not os.path.isfile(report):
        st.info("No evaluation results found. Run `python evaluate.py --dataset .` first.")
        return

    kv = {}
    with open(report) as f:
        for line in f:
            if ":" in line:
                k, _, v = line.partition(":")
                try:
                    kv[k.strip()] = float(v.strip())
                except ValueError:
                    kv[k.strip()] = v.strip()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{kv.get('Accuracy', 0):.2%}")
    c2.metric("Precision", f"{kv.get('Precision', 0):.2%}")
    c3.metric("Recall", f"{kv.get('Recall', 0):.2%}")
    c4.metric("F1 Score", f"{kv.get('F1 Score', 0):.2%}")
    c5.metric("AUC-ROC", f"{kv.get('AUC-ROC', 0):.4f}")

    col_a, col_b = st.columns(2)
    if os.path.isfile(cm_img):
        with col_a:
            st.image(cm_img, width=400)
    if os.path.isfile(roc_img):
        with col_b:
            st.image(roc_img, width=400)

    if os.path.isfile(dist_img):
        st.image(dist_img, width=700)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def build_sidebar():
    opt_thr = _load_optimal_threshold()

    with st.sidebar:
        st.markdown("## 🎭 DeepFake Detector")
        st.caption("EfficientNetB2 + LSTM")
        st.markdown("---")

        st.markdown("### ⚙️ Decision Threshold")

        if opt_thr is not None:
            st.success(f"✅ Calibrated: **{opt_thr:.4f}**")
            default_thr = opt_thr
        else:
            st.info("Run `python evaluate.py --dataset .` to calibrate")
            default_thr = 0.50

        threshold = st.slider(
            "Decision threshold",
            min_value=0.20,
            max_value=0.80,
            value=float(round(default_thr, 2)),
            step=0.01,
            help="Score ≥ threshold → FAKE"
        )

        show_dbg = st.checkbox("Show debug panel", value=False)

        st.markdown("---")
        st.markdown(f"**Current threshold:** {threshold:.2f}")
        st.markdown(f"- Score **≥ {threshold:.2f}** → 🚨 FAKE")
        st.markdown(f"- Score **< {threshold:.2f}** → ✅ REAL")

    return threshold, show_dbg


# ═════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS TAB
# ═════════════════════════════════════════════════════════════════════════════
def render_how_it_works():
    st.subheader("🎓 How It Works — Viva Preparation")

    st.markdown("""
### Step-by-step pipeline

| # | Step | What happens | Why |
|---|------|-------------|-----|
| 1 | Upload | Raw image / video enters the system | — |
| 2 | Face detection | Haar cascade isolates the face region | Deepfakes alter faces, not backgrounds |
| 3 | 25 variants | Crop × brightness grid creates a sequence | LSTM expects temporal sequence |
| 4 | Feature extraction | EfficientNetB2 → 1408-dim embedding | Pre-trained CNN captures artifacts |
| 5 | LSTM classification | Processes sequence → raw score | Temporal model detects inconsistencies |
| 6 | Threshold | Score ≥ threshold → FAKE | Optimized using Youden's Index |

### How to read the results

| Raw Score | Verdict | Meaning |
|-----------|---------|---------|
| 0.00 - 0.30 | REAL | Media appears authentic |
| 0.30 - 0.60 | Uncertain | Borderline - manual review recommended |
| 0.60 - 1.00 | FAKE | Manipulation artifacts detected |
""")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown('<p class="title-text">🎭 DeepFake Detection System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">EfficientNetB2 + LSTM · Threshold-calibrated</p>',
                unsafe_allow_html=True)

    threshold, show_dbg = build_sidebar()

    with st.spinner("🔄 Loading AI models…"):
        clf, backbone, err = load_models()

    if err or clf is None:
        st.error(f"❌ Could not load model: {err}")
        st.stop()

    st.success("✅ Models loaded successfully.")
    st.markdown("---")

    tab_detect, tab_eval, tab_how = st.tabs(["🔍 Detection", "📊 Evaluation Results", "🎓 How It Works"])

    with tab_detect:
        mode = st.radio("Select input type:", ["📷 Image", "🎥 Video"], horizontal=True)
        st.markdown("---")

        if mode == "📷 Image":
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if up is not None:
                raw_bytes = up.read()
                img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

                col_img, col_face = st.columns(2)
                with col_img:
                    st.image(to_rgb(img_bgr), caption="Uploaded image", width=400)

                if st.button("🔍 Analyse Image", use_container_width=True):
                    with st.spinner("Analysing…"):
                        label, conf, raw, band, face, dbg = analyze_image(
                            clf, backbone, img_bgr, threshold)

                    with col_face:
                        if face is not None:
                            st.image(to_rgb(face), caption="Detected face", width=400)
                        else:
                            st.info("ℹ️ No face detected")

                    st.markdown("---")
                    show_verdict(label, conf, raw, band, threshold)
                    render_insights(dbg)
                    if show_dbg:
                        render_debug(dbg)

        else:  # Video mode
            up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
            if up is not None:
                video_bytes = up.read()
                st.video(video_bytes)

                if st.button("🔍 Analyse Video", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_bytes)
                        tmp_path = tmp.name

                    prog_ph = st.empty()
                    try:
                        with st.spinner("Analysing video…"):
                            (label, conf, raw, band, frame_scores,
                             frames, face_boxes, dbg) = analyze_video(
                                clf, backbone, tmp_path, threshold, progress_ph=prog_ph)
                    finally:
                        prog_ph.empty()
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                    st.markdown("---")
                    show_verdict(label, conf, raw, band, threshold)
                    render_insights(dbg)

                    if frame_scores:
                        st.markdown("---")
                        st.subheader("📊 Frame-by-frame analysis")
                        plot_frame_scores(frame_scores, threshold)

                    if show_dbg:
                        render_debug(dbg)

    with tab_eval:
        render_eval_tab()

    with tab_how:
        render_how_it_works()


if __name__ == "__main__":
    main()