import os, glob, textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER

IN = "data/corpus_txt"
OUT = "data/generated_pdfs/reportlab"
os.makedirs(OUT, exist_ok=True)

for txt_path in sorted(glob.glob(f"{IN}/doc_*.txt")):
    base = os.path.splitext(os.path.basename(txt_path))[0] 
    pdf_path = f"{OUT}/{base}.pdf"
    with open(txt_path, "r") as f:
        body = f.read()

    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    w, h = LETTER
    lines = body.splitlines()
    y = h - 110
    c.setFont("Times-Roman", 11)
    for line in lines:
        c.drawString(72, y, line)
        y -= 14
        if y < 72:
            c.showPage()
            c.setFont("Times-Roman", 11)
            y = h - 110
    c.save()
print("ReportLab PDFs ready.")
