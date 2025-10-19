# Forensics – Renderer Provenance Detection

This project identifies which software generated a PDF (LibreOffice or ReportLab) purely from visual features extracted from rendered images. The data is uploaded in the ub box and the link is shared.
It demonstrates that even when text and formatting are identical, distinct rasterization patterns, font rendering, and layout cues reveal the document’s provenance.



## Objective

To build a fully automated, leak-free pipeline that:
- Generates synthetic text documents.
- Converts them into PDFs using two renderers (LibreOffice and ReportLab).
- Renders all PDFs to images.
- Extracts features (HOG, histogram, statistics).
- Trains multiple classifiers (SVM, SGD, RF, MLP) to detect the source renderer.

---

## Pipeline Overview

| Stage | Description |
|--------|--------------|
| Text Generation | Builds a synthetic corpus of realistic English paragraphs. |
| DOCX Creation | Converts text → DOCX with a uniform Times New Roman 11 pt font. |
| PDF Generation (LibreOffice) | Uses `soffice` CLI to export DOCX → PDF. |
| PDF Generation (ReportLab) | Directly generates PDFs from text via the ReportLab API. |
| PDF → Image Conversion | Renders each PDF to PNG using `PyMuPDF`. |
| Feature Extraction & Classification | Trains models using HOG + intensity statistics. |

---

## Setup

See [setup.md](./setup.md) for detailed environment setup and installation instructions.

Quick summary:
terminal:
python -m venv venv
source venv/bin/activate        # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

## Checks
[Leak check] Unique train docs: 16000
[Leak check] Unique test docs: 4000
[Leak check] Overlap: 0
[Sanity] Accuracy with SHUFFLED training labels: 0.4905


## Structure
src/
  make_corpus_texts.py          # Generate text corpus
  docx_from_corpus.py           # Create DOCX files
  pdf_generation_libreoffice.py # Convert DOCX -> PDF
  gen_reportlab_pdfs.py         # Generate ReportLab PDFs
  pdf_to_images.py              # Render PDFs -> PNGs
  classification_from_subfolders.py # Train classifiers
  ablation_make_textless_images.py  # Optional blur ablation
  cross_domain_eval.py              # Cross-domain testing
data/
  corpus_txt/  docx/  generated_pdfs/{libre,reportlab}  images/{libre,reportlab}
results/
  models/



