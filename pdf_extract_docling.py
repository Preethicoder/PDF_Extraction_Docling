from docling.document_converter import DocumentConverter
from pathlib import Path

import fitz
from pathlib import Path
from docling.document_converter import DocumentConverter
import fitz
import pandas as pd

def extract_pdf_docling(pdf_path: Path, output_dir: Path, extract_images=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pdf_path.stem

    converter = DocumentConverter()
    res = converter.convert(pdf_path)
    doc = res.document

    text_file = output_dir / f"{base}.txt"
    text_file.write_text(doc.export_to_text(), encoding="utf-8")

    for idx, tbl in enumerate(doc.tables, 1):
        df = tbl.export_to_dataframe()
        df.to_csv(output_dir / f"{base}-table-{idx}.csv", index=False)

    img_count = 0
    if extract_images:
        pdf_doc = fitz.open(pdf_path)
        for page_index in range(min(len(pdf_doc), 3)):  # Limit to 3 pages for dev
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


# Step 1: Create a list of PDFs
pdf_folder = Path("sample_pdf")
pdf_files = list(pdf_folder.glob("*.pdf"))

# Step 2: Loop over each file
#for pdf_file in pdf_files:
    #print(f"🔄 Starting: {pdf_file.name}")
extract_pdf_docling(Path("/Users/preethisivakumar/Documents/spacelyout/sample_pdf/gesamtes LV.pdf"), Path("output"))
