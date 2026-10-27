
from document_loader import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read the PDF
pages = extract_text_from_pdf("documents/sample.pdf")

# Combine text from all pages
all_text = "\n".join(page["text"] for page in pages)

# Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(all_text)

# Display the results
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
