
import pymupdf
from pathlib import Path


def load_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Returns:
        A list of dictionaries containing
        page content and metadata.
    """

    pdf_path = Path(pdf_path)

    # Check whether the file exists
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    documents = []

    # Open PDF
    pdf = pymupdf.open(pdf_path)

    try:
        for page_number, page in enumerate(pdf):

            # Extract text from current page
            text = page.get_text("text").strip()

            # Skip pages without extractable text
            if not text:
                continue

            documents.append({
                "page_content": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number + 1
                }
            })

    finally:
        pdf.close()

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF."
        )

    return documents

if __name__ == "__main__":

    pdf_path = "data/documents/R1_TestDocument.pdf"

    documents = load_pdf(pdf_path)

    print("Total pages extracted:", len(documents))

    print("\nFirst page metadata:")
    print(documents[0]["metadata"])

    print("\nExtracted text:")
    print(documents[0]["page_content"][:500])