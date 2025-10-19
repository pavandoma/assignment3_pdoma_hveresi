create virtualenv
python3 -m venv .venv && source .venv/bin/activate pip install -r requirements.txt

install libreoffice (for PDF conversions)
sudo apt-get update && sudo apt-get install -y libreoffice libreoffice-writer poppler-utils
project_root/
├─ src/
│  ├─ make_corpus_texts.py
│  ├─ docx_from_corpus.py
│  ├─ pdf_generation_libreoffice.py
│  ├─ gen_reportlab_pdfs.py
│  ├─ pdf_to_images.py
│  ├─ classification_from_subfolders.py
│  ├─ ablation_make_textless_images.py
│  └─ cross_domain_eval.py
├─ data/
│  ├─ corpus_txt/
│  ├─ docx/
│  ├─ generated_pdfs/
│  │  ├─ libre/
│  │  └─ reportlab/
│  └─ images/
│     ├─ libre/
│     └─ reportlab/
├─ results/
│  └─ models/
├─ setup.md
└─ README.md

# Step 1: Generate synthetic text files
python src/make_corpus_texts.py

# Step 2: Convert text → DOCX (same font and layout)
python src/docx_from_corpus.py

# Step 3: Convert DOCX → PDFs using LibreOffice
python src/pdf_generation_libreoffice.py

# Step 4: Generate PDFs using ReportLab
python src/gen_reportlab_pdfs.py

# Step 5: Convert PDFs → Images (for both sources)
python src/pdf_to_images.py --input_dir data/generated_pdfs/libre     --output_dir data/images/libre
python src/pdf_to_images.py --input_dir data/generated_pdfs/reportlab --output_dir data/images/reportlab

# Step 6: Train classifiers and view results
python src/classification_from_subfolders.py
