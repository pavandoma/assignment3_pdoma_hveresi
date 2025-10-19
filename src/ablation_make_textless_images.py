import os, cv2, glob
from pathlib import Path

SRC_ROOT = "data/images"          # current images
DST_ROOT = "data/images_textless" # output here
BLUR = 9                          # try 7, 9, 11

for cls in sorted(os.listdir(SRC_ROOT)):
    src = Path(SRC_ROOT, cls)
    dst = Path(DST_ROOT, cls)
    if not src.is_dir(): continue
    dst.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(str(src / "*.png")):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        out = cv2.GaussianBlur(img, (BLUR, BLUR), 0)  # wipe glyph detail
        cv2.imwrite(str(dst / Path(p).name), out)

print("Textless images written to", DST_ROOT)
