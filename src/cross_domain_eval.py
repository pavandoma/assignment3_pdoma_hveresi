import os, cv2, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

IMG_SIZE = (96,96)

def extract_features(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    mean,std = float(np.mean(img)), float(np.std(img))
    q25,q50,q75 = [float(x) for x in np.percentile(img,[25,50,75])]
    cell=8; block=16; stride=8
    hog = cv2.HOGDescriptor(_winSize=IMG_SIZE,_blockSize=(block,block),
                            _blockStride=(stride,stride),_cellSize=(cell,cell),_nbins=9)
    try: h = hog.compute(img).flatten().astype(np.float32)
    except: h = np.zeros((hog.getDescriptorSize(),), np.float32)
    hist = cv2.calcHist([img],[0],None,[32],[0,256]).flatten().astype(np.float32)
    s = float(np.sum(hist))
    if s > 0: hist /= s
    feats = np.concatenate(([mean,std,q25,q50,q75],h,hist)).astype(np.float32)
    return feats

def load(folder):
    X=[]
    for p in Path(folder).glob("*.png"):
        f = extract_features(str(p))
        if f is not None: X.append(f)
    return np.array(X,np.float32)

TRAIN_DIR = "data/images/libre"
TEST_DIR  = "data/images/reportlab"


Xtr=load(TRAIN_DIR); ytr=np.zeros(len(Xtr),dtype=int)
Xte=load(TEST_DIR);  yte=np.ones(len(Xte),dtype=int)

sc = StandardScaler(with_mean=False)
Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

clf = SVC(kernel="linear", random_state=42)
clf.fit(Xtr,ytr)
pred = clf.predict(Xte)

print(f"Train={Path(TRAIN_DIR).name} → Test={Path(TEST_DIR).name}")
print("Accuracy:", accuracy_score(yte,pred))
print(classification_report(yte,pred,target_names=['libre','reportlab']))
