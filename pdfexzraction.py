import spacy
from spacy_layout import spaCyLayout

nlp = spacy.load("en_core_web_sm")

layout = spaCyLayout(nlp)

doc = layout("/Users/preethisivakumar/Documents/spacelyout/sample_pdf/gesamtes LV.pdf")

#print(doc.text)
print(doc._.markdown)
#print(doc._.tables)
for table in doc._.tables:
    print(table.start, table.end, table._.layout)
    print(table._.data)

"""import pandas as pd
def display_table(df: pd.DataFrame):
    return f"Table with columns: {', '.join(df.columns.tolist())}"

layout = spaCyLayout(nlp, display_table=display_table)
doc = layout("/Users/preethisivakumar/Documents/spacelyout/sample_pdf/arztbrief_innere_medizin.pdf")

print(doc)
#print(doc._.layout)"""
#doc._.layout.pages
# Import required libraries
import pypdfium2 as pdfium
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load and convert the PDF page to an image
"""pdf = pdfium.PdfDocument("/Users/preethisivakumar/Documents/spacelyout/Laborbefund.pdf")
page_image = pdf[2].render(scale=1)  # Get page 3 (index 2)
numpy_array = page_image.to_numpy()

# Get page 3 layout and sections
page = doc._.pages[2]
page_layout = doc._.layout.pages[2]

# Create figure and axis with page dimensions
fig, ax = plt.subplots(figsize=(12, 16))

# Display the PDF image
ax.imshow(numpy_array)

# Add rectangles for each section's bounding box
for section in page[1]:
    layout = section._.layout
    # Create rectangle patch
    rect = Rectangle(
        (layout.x, layout.y),
        layout.width,
        layout.height,
        fill=False,
        color='blue',
        linewidth=1,
        alpha=0.5
    )
    ax.add_patch(rect)

    # Add text label at top of box
    ax.text(layout.x, layout.y, section.label_,
            fontsize=8, color='red',
            verticalalignment='bottom')

# Set title and display
ax.set_title('Page 3 Layout with Bounding Boxes')
ax.axis('off')  # Hide axes
plt.show()"""

