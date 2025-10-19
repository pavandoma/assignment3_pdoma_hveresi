import os
import argparse
import fitz
from tqdm import tqdm

def convert_pdfs_to_images(input_dir, output_dir, zoom=2):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    for pdf_file in tqdm(pdf_files, desc=f"Converting PDFs → Images [{os.path.basename(input_dir)}]"):
        pdf_path = os.path.join(input_dir, pdf_file)
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            out = os.path.join(
                output_dir,
                f"{os.path.splitext(pdf_file)[0]}_page{page_num+1}.png"
            )
            pix.save(out)
        doc.close()
    print(f"Done: {input_dir} → {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--zoom", type=int, default=2)
    args = ap.parse_args()
    convert_pdfs_to_images(args.input_dir, args.output_dir, zoom=args.zoom)
