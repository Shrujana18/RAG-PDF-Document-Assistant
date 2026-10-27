from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """
    Extract text from every page of a PDF.

    Returns a list containing:
    - text
    - source document name
    - page number
    """

    documents = []

    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Skip empty pages
        if not text or not text.strip():
            continue

        documents.append({
            "text": text.strip(),
            "source": "sample.pdf",
            "page": page_number
        })

    return documents