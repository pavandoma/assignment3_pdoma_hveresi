
import os, cv2, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


IMG_SIZE = (128, 128) 
SAVE_MODELS = True  
OUT_DIR = "results/models"

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None

    IMG_W, IMG_H = IMG_SIZE
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32)

    mean = float(np.mean(img_f))
    std  = float(np.std(img_f))
    q25, q50, q75 = [float(x) for x in np.percentile(img_f, [25, 50, 75])]

    cell  = 16 if IMG_W >= 128 else 8
    block = cell * 2
    stride = cell
    hog = cv2.HOGDescriptor(
        _winSize=(IMG_W, IMG_H),
        _blockSize=(block, block),
        _blockStride=(stride, stride),
        _cellSize=(cell, cell),
        _nbins=9
    )
    try:
        h = hog.compute(img).flatten().astype(np.float32)
    except Exception:
        h = np.zeros((hog.getDescriptorSize(),), dtype=np.float32)

    hist = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten().astype(np.float32)
    hist_sum = float(np.sum(hist))
    if hist_sum > 0:
        hist /= hist_sum
    else:
        hist[:] = 0.0

    feats = np.concatenate((
        np.array([mean, std, q25, q50, q75], dtype=np.float32),
        h,
        hist
    )).astype(np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats

def load_dataset(root="data/images"):
    X, y, groups, classes = [], [], [], []
    classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
    label_map = {c:i for i,c in enumerate(classes)}

    for cls in classes:
        for p in Path(root, cls).glob("*.png"):
            feats = extract_features(str(p))
            if feats is None: continue
            X.append(feats)
            y.append(label_map[cls])
            stem = Path(p).stem          
            doc_id = stem.split("_page")[0] 
            groups.append(f"{cls}::{doc_id}") 
    return np.array(X, np.float32), np.array(y, np.int64), np.array(groups), classes

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, groups, classes = load_dataset("data/images")
    if len(X) == 0:
        raise SystemExit("No images found. Did you run pdf_to_images.py per source?")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    docs_tr = set(groups[train_idx])
    docs_te = set(groups[test_idx])
    overlap = docs_tr.intersection(docs_te)
    print(f"\n[Leak check] Unique train docs: {len(docs_tr)}, test docs: {len(docs_te)}, overlap: {len(overlap)} (should be 0)")

    mask = np.array([np.isfinite(x).all() for x in X], dtype=bool)
    if not mask.all():
        bad = np.where(~mask)[0]
        print(f"[Warn] Dropping {len(bad)} samples with non-finite features.")
        X, y, groups = X[mask], y[mask], groups[mask]

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


    scaler = StandardScaler(with_mean=False)
    Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)

    models = {
        "SVM": SVC(kernel="linear", random_state=42),
        "SGD": SGDClassifier(random_state=42, max_iter=2000, tol=1e-3),
        "RF":  RandomForestClassifier(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,128), max_iter=200, random_state=42)
    }

    rows=[]
    for name, m in models.items():
        print(f"\nTraining {name}...")
        m.fit(Xtr, ytr)
        yhat = m.predict(Xte)
        acc = accuracy_score(yte, yhat)
        print(f"{name} acc: {acc:.4f}")
        print(classification_report(yte, yhat, target_names=classes))
        if SAVE_MODELS:
            dump(m, f"{OUT_DIR}/{name}_model.joblib")
        rows.append({"model": name, "accuracy": acc})

    dump(scaler, f"{OUT_DIR}/scaler.joblib")
    with open(f"{OUT_DIR}/label_map.json", "w") as f:
        json.dump({c:i for i,c in enumerate(classes)}, f, indent=2)

    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/metrics.csv", index=False)
    print(f"Saved metrics (and models if enabled) to {OUT_DIR}")

#ytr_shuf = ytr.copy()
#rng = np.random.default_rng(42)
#rng.shuffle(ytr_shuf)
#chk = RandomForestClassifier(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42)
#chk.fit(Xtr, ytr_shuf)
#acc_shuf = accuracy_score(yte, chk.predict(Xte))
#print(f"\n[Sanity] Accuracy with SHUFFLED training labels: {acc_shuf:.4f}")
