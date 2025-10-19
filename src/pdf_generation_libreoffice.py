import os, sys, glob, shutil, subprocess
from pathlib import Path

DOCX_DIR = Path("data/docx").resolve()
OUT_DIR  = Path("data/generated_pdfs/libre").resolve()

BATCH = 150

def find_soffice():
    p = shutil.which("soffice")
    if p: return p

    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.com",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.com",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

def main():
    if not DOCX_DIR.exists():
        print(f"ERROR: DOCX_DIR not found: {DOCX_DIR}")
        sys.exit(1)

    docx_files = sorted(str(p) for p in DOCX_DIR.glob("doc_*.docx"))
    if not docx_files:
        print(f"No DOCX found in {DOCX_DIR}. Run docx_from_corpus.py first.")
        sys.exit(1)

    soffice = find_soffice()
    if not soffice:
        print("ERROR: Could not find LibreOffice 'soffice'.")
        print(r"Install it and/or set the path, e.g. C:\Program Files\LibreOffice\program\soffice.com")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using: {soffice}")
    print(f"Converting {len(docx_files)} DOCX → PDF into {OUT_DIR} (batch={BATCH})")

    for i in range(0, len(docx_files), BATCH):
        chunk = docx_files[i:i+BATCH]
        cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT_DIR)]
        cmd += chunk
        print(f"[{i:05d}..{i+len(chunk)-1:05d}] Converting {len(chunk)} files...")
        subprocess.run(cmd, check=True)

    print(f"Conversion complete! PDFs saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
