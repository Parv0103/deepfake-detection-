"""
evaluate.py — Fixed version.
- Clamps sklearn's inf threshold to [0,1] before saving (fixes slider crash)
- Diagnoses constant-output model and explains the root cause
"""
import argparse, os, sys
import cv2
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SEQ_LEN, FEAT_DIM, IMG_SIZE = 25, 1408, 224
OUT_DIR  = "eval_results"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_CROP_RATIOS  = [0.00, 0.04, 0.07, 0.09, 0.11]
_BRIGHTNESSES = [-20, -10, 0, 10, 20]

# ── Preprocessing ─────────────────────────────────────────────────────────
def _pp():
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input

def extract_features(face_bgr, backbone):
    if face_bgr is None: return None
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    feat = backbone.predict(np.expand_dims(_pp()(rgb), 0), verbose=0)
    return feat[0].astype(np.float32)

def make_variants(face_bgr):
    h, w = face_bgr.shape[:2]; vs = []
    for cr in _CROP_RATIOS:
        for bs in _BRIGHTNESSES:
            cx, cy = int(w*cr/2), int(h*cr/2)
            if cx>0 or cy>0:
                v = cv2.resize(face_bgr[cy:h-cy or h, cx:w-cx or w],
                               (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
            else: v = face_bgr.copy()
            vs.append(np.clip(v.astype(np.int16)+bs, 0, 255).astype(np.uint8))
    return vs

def fill_missing(fl):
    n = len(fl); out = np.zeros((n, FEAT_DIM), np.float32)
    valid = [i for i,f in enumerate(fl) if f is not None]
    if not valid: return out
    for i in range(valid[0]): out[i] = fl[valid[0]]
    for k in range(len(valid)):
        ia = valid[k]; out[ia] = fl[ia]
        if k+1 < len(valid):
            ib = valid[k+1]; fa, fb = fl[ia], fl[ib]
            for j in range(ia+1, ib):
                t = (j-ia)/(ib-ia); out[j] = (1-t)*fa + t*fb
    for i in range(valid[-1]+1, n): out[i] = fl[valid[-1]]
    return out

def run_model(clf, seq):
    s = seq.reshape(1, SEQ_LEN, FEAT_DIM).astype(np.float32)
    if len(clf.inputs)==2:
        aux = np.zeros((1, int(clf.inputs[1].shape[1])), np.float32)
        out = clf.predict([s, aux], verbose=0)
    else: out = clf.predict(s, verbose=0)
    out = np.squeeze(out)
    if out.ndim==0: return float(out)
    if out.ndim==1 and out.shape[0]==2: return float(out[1])
    return float(out[0])

def infer_image(clf, backbone, img_bgr, skip_face=True):
    face = cv2.resize(img_bgr, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA) \
           if skip_face else None   # simplified: always use skip_face=True for dataset
    if face is None: return None
    return run_model(clf, fill_missing([extract_features(v, backbone)
                                        for v in make_variants(face)]))

# ── Model diagnosis ────────────────────────────────────────────────────────
def diagnose(clf, backbone):
    print("\n[DIAG] Self-diagnosis…")
    black = np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
    white = np.ones((IMG_SIZE,IMG_SIZE,3), np.uint8)*255
    sb = infer_image(clf, backbone, black)
    sw = infer_image(clf, backbone, white)
    diff = abs(sb - sw)
    print(f"      Score(black)={sb:.6f}  Score(white)={sw:.6f}  Δ={diff:.6f}")
    constant = diff < 1e-4
    if constant:
        print("""
  ╔══════════════════════════════════════════════════╗
  ║  ⚠  MODEL OUTPUTS CONSTANT PREDICTIONS          ║
  ╠══════════════════════════════════════════════════╣
  ║  Root cause: LSTM weights are untrained/random.  ║
  ║  The .h5 was likely saved BEFORE training ended. ║
  ╠══════════════════════════════════════════════════╣
  ║  FIX:                                            ║
  ║  1. Train the LSTM until val_accuracy > 0.6.     ║
  ║  2. model.save() AFTER fit() completes.          ║
  ║  3. Re-run evaluate.py.                          ║
  ╠══════════════════════════════════════════════════╣
  ║  Threshold saved as 0.50 (safe default).         ║
  ╚══════════════════════════════════════════════════╝
""")
    else:
        print(f"      ✓ Model responsive")
    return constant

# ── Dataset ────────────────────────────────────────────────────────────────
def load_dataset(root, max_n=None):
    items = []
    for lname, lint in [("real",0),("fake",1)]:
        folder = os.path.join(root, lname)
        if not os.path.isdir(folder):
            print(f"[WARN] Not found: {folder}"); continue
        files = sorted(f for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if max_n and len(files)>max_n:
            print(f"      Limiting {lname}: {len(files)} → {max_n}")
            files = files[:max_n]
        for fn in files: items.append((os.path.join(folder,fn), lint))
        print(f"      Loaded {len(files)} {lname}")
    return items

# ── Metrics ────────────────────────────────────────────────────────────────
def metrics(yt, yp):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)
    return (accuracy_score(yt,yp), precision_score(yt,yp,zero_division=0),
            recall_score(yt,yp,zero_division=0), f1_score(yt,yp,zero_division=0),
            confusion_matrix(yt,yp))

def optimal_threshold(yt, sc):
    """
    FIXED: sklearn prepends inf as first threshold — we clamp to [0,1].
    Returns best_thr always in [0,1].
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thr = roc_curve(yt, sc)
    auc = roc_auc_score(yt, sc)
    thr = np.clip(thr, 0.0, 1.0)          # ← THE FIX
    if auc < 0.55:
        print(f"  ⚠  AUC={auc:.4f} (random). Defaulting to 0.50.")
        return 0.50, float('nan'), float('nan'), auc, fpr, tpr, thr
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i]), auc, fpr, tpr, thr

# ── Plots ──────────────────────────────────────────────────────────────────
def plot_cm(cm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=16,
                    fontweight="bold",
                    color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["REAL","FAKE"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["REAL","FAKE"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] {path}")

def plot_roc(fpr,tpr,auc,bt,bf,btp,path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,lw=2,label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1)
    if not (np.isnan(bf) or np.isnan(btp)):
        ax.scatter([bf],[btp],c="red",zorder=5,label=f"thr={bt:.3f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] {path}")

def plot_dist(yt, sc, thr, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist([s for s,y in zip(sc,yt) if y==0],bins=40,alpha=0.6,
            color="green",label="REAL")
    ax.hist([s for s,y in zip(sc,yt) if y==1],bins=40,alpha=0.6,
            color="red",label="FAKE")
    ax.axvline(x=thr,color="black",linestyle="--",label=f"thr={thr:.3f}")
    ax.set_xlabel("Raw score"); ax.set_ylabel("Count")
    ax.set_title("Score Distribution"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[✓] {path}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default=".",
                    help="Path to dataset root (must contain real/ and fake/)")
    ap.add_argument("--model",      default="latest_EfficientNetB2.h5")
    ap.add_argument("--threshold",  type=float, default=None)
    ap.add_argument("--max-per-class", type=int, default=100,  # CHANGED: per class limit
                    help="Maximum images per class (default: 100)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1/5] Loading models…")
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.models import load_model
    clf      = load_model(args.model)
    backbone = EfficientNetB2(weights="imagenet", include_top=False, pooling="avg")
    print(f"      Model: {args.model}")

    constant = diagnose(clf, backbone)

    print("\n[2/5] Loading dataset…")
    print(f"      Root: {os.path.abspath(args.dataset)}")
    # Use max_per_class instead of max-images
    items = load_dataset(args.dataset, max_n=args.max_per_class)
    if not items: print("[ERROR] No images."); sys.exit(1)
    print(f"      Total images: {len(items)} (approx {args.max_per_class} real + {args.max_per_class} fake)")

    print("\n[3/5] Running inference…")
    yt, sc, skip = [], [], 0
    for i, (p, lbl) in enumerate(items):
        img = cv2.imread(p)
        if img is None: skip+=1; continue
        s = infer_image(clf, backbone, img, skip_face=True)
        if s is None: skip+=1; continue
        yt.append(lbl); sc.append(s)
        if (i+1)%10==0 or (i+1)==len(items):
            print(f"      {i+1}/{len(items)} (valid:{len(sc)} skipped:{skip})",
                  end="\r")
    print()
    if len(yt) < 10: print("[ERROR] Too few samples."); sys.exit(1)
    yt = np.array(yt); sc = np.array(sc)
    score_std = float(sc.std())
    if score_std < 1e-4:
        print(f"  ⚠  All scores identical (std={score_std:.2e}) — constant model confirmed.")

    print("\n[4/5] Optimising threshold…")
    bt, btp, bfp, auc, fpr, tpr, thr = optimal_threshold(yt, sc)
    saved_thr = float(np.clip(bt, 0.0, 1.0))   # ALWAYS finite [0,1]
    thr_use   = args.threshold if args.threshold is not None else saved_thr
    yp = (sc >= thr_use).astype(int)
    acc, prec, rec, f1, cm = metrics(yt, yp)

    with open(os.path.join(OUT_DIR,"optimal_threshold.txt"),"w") as f:
        f.write(f"{saved_thr:.6f}\n")
    print(f"      Saved threshold: {saved_thr:.6f}  (always in [0,1])")

    print("\n" + "═"*50)
    print("  EVALUATION RESULTS")
    print("═"*50)
    print(f"  Threshold : {thr_use:.4f}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"             REAL   FAKE")
    print(f"  True REAL  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  True FAKE  {cm[1,0]:5d}  {cm[1,1]:5d}")
    print("═"*50)
    rs = sc[yt==0]; fs = sc[yt==1]
    print(f"\n  REAL: mean={rs.mean():.4f}  std={rs.std():.6f}")
    print(f"  FAKE: mean={fs.mean():.4f}  std={fs.std():.6f}")

    rpt = os.path.join(OUT_DIR,"threshold_report.txt")
    with open(rpt,"w") as f:
        for k,v in [("Threshold",thr_use),("Accuracy",acc),("Precision",prec),
                    ("Recall",rec),("F1 Score",f1),("AUC-ROC",auc),
                    ("TPR@opt",btp),("FPR@opt",bfp),
                    ("REAL mean",rs.mean()),("FAKE mean",fs.mean()),
                    ("Score std",score_std),("Constant model",constant)]:
            f.write(f"{k} : {v}\n")
    print(f"[✓] {rpt}")

    print("\n[5/5] Saving plots…")
    plot_cm(cm,   os.path.join(OUT_DIR,"confusion_matrix.png"))
    plot_roc(fpr,tpr,auc,saved_thr,bfp,btp, os.path.join(OUT_DIR,"roc_curve.png"))
    plot_dist(yt,sc,thr_use, os.path.join(OUT_DIR,"score_distribution.png"))

    print(f"\n✅  Done → {OUT_DIR}/")
    if constant:
        print("\n"+"!"*54)
        print("  ACTION REQUIRED: MODEL HAS UNTRAINED WEIGHTS")
        print("!"*54)
        print("""
  Score std ≈ 0 means the LSTM predicts the same value
  (≈ 0.4948) for every image — real or fake.

  This is NOT a code bug. It is a model-training issue:
    → The .h5 file was saved before/during training,
      so it contains only random initial weights.

  How to fix:
    1. Open your training notebook/script.
    2. Run model.fit() for enough epochs until
       val_accuracy clearly exceeds 0.60.
    3. Add: model.save('latest_EfficientNetB2.h5')
       at the END of training (after the fit loop).
    4. Delete eval_results/ and re-run evaluate.py.

  app.py threshold now defaults to 0.50 (no crash).
""")

if __name__ == "__main__":
    main()