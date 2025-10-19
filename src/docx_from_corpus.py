import os, glob
from docx import Document

IN_DIR = "data/corpus_txt"
OUT_DIR = "data/docx"
os.makedirs(OUT_DIR, exist_ok=True)

for txt_path in sorted(glob.glob(f"{IN_DIR}/doc_*.txt")):
    base = os.path.splitext(os.path.basename(txt_path))[0] 
    doc = Document()
    with open(txt_path, "r") as f:
        content = f.read()
    for line in content.splitlines():
        doc.add_paragraph(line)
    doc.save(f"{OUT_DIR}/{base}.docx")

print(f"DOCX written to {OUT_DIR}")
