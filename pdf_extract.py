from docling.document_converter import DocumentConverter
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF for extracting embedded images
"""  layout analysis (like DocLayNet) and table structure recognition (like TableFormer).
model prefetching and offline usage"""
from docling.document_converter import DocumentConverter
from pathlib import Path

import fitz  # PyMuPDF for extracting embedded images

def extract_pdf_docling(pdf_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pdf_path.stem

    # Step 1: Run Docling with OCR installed (auto OCR fallback)
    converter = DocumentConverter()
    res = converter.convert(pdf_path)
    doc = res.document

    # Save extracted text
    text_file = output_dir / f"{base}.txt"
    text_file.write_text(doc.export_to_text(), encoding="utf-8")

    # Save extracted tables
    for idx, tbl in enumerate(doc.tables, 1):
        df = tbl.export_to_dataframe()
        df.to_csv(output_dir / f"{base}-table-{idx}.csv", index=False)

    # Step 2: Extract embedded images
    pdf_doc = fitz.open(pdf_path)
    img_count = 0
    for page_index in range(len(pdf_doc)):
        for img_index, img in enumerate(pdf_doc.get_page_images(page_index)):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            img_ext = base_image["ext"]
            image_bytes = base_image["image"]
            img_count += 1
            img_file = output_dir / f"{base}-image-{img_count}.{img_ext}"
            with open(img_file, "wb") as f:
                f.write(image_bytes)

    print(f"✅ {pdf_path.name}: {len(doc.tables)} tables, {img_count} images, text saved to {text_file}")

# Run on all PDFs in the 'sample_pdf' folder
def process_all_pdfs_in_folder(pdf_folder: str, output_dir: str):
    pdf_folder = Path(pdf_folder)
    output_dir = Path(output_dir)

    for pdf_file in pdf_folder.glob("*.pdf"):
        extract_pdf_docling(pdf_file, output_dir)

# Example usage
process_all_pdfs_in_folder("sample_pdf", "output")
