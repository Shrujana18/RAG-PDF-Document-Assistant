
from document_loader import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, search_documents

# Step 1: Read the PDF
pages = extract_text_from_pdf("documents/sample.pdf")

# Step 2: Combine the text
all_text = "\n".join(page["text"] for page in pages)

# Step 3: Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(all_text)

if not chunks:
    print("No text found in the PDF.")
else:
    # Step 4: Create the searchable store
    print("Creating searchable store...")
    model, index, stored_chunks = create_vector_store(chunks)

    # Step 5: Ask a question
    question = input("\nAsk a question about the PDF: ")

    # Step 6: Search for relevant text
    results = search_documents(
        question, model, index, stored_chunks
    )

    print("\nRelevant information:")
    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(result)
