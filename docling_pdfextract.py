from docling.document_converter import DocumentConverter

"""Just need document conversion? → Use Docling
Need document conversion + AI analysis? → Use SpacyLayout
Already using spaCy for NLP? → Use SpacyLayout

SpacyLayout is basically Docling + spaCy NLP combined into one tool."""
def docling_example():
    """
    Docling: Converts documents to structured format
    Focus: Document parsing and conversion
    """
    converter = DocumentConverter()
    result = converter.convert("/Users/preethisivakumar/Documents/spacelyout/Laborbefund.pdf")

    # What you get from Docling:
    doc = result.document

    # 1. Raw text extraction
    text = doc.export_to_markdown()
    json_format = doc.export_to_dict()
    print("DOCLING OUTPUT:")


    # 2. Structured data
    tables = doc.tables  # Structured table data
    images = doc.pictures  # Image objects

    print(text)
    for i, table in enumerate(tables):
        #print(table.start, table.end, table.layout)
        #print(table.data)
        df = table.export_to_dataframe()
        df.to_csv(f"table_{i + 1}.csv", index=False)
        df.to_excel(f"table_{i + 1}.xlsx", index=False)



    # What you get from Docling:

docling_example()