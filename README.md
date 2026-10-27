# PDF Document Assistant Using RAG

## 1. Project Overview
The PDF Document Assistant is an AI-powered application that allows users to upload PDF documents and ask questions about their content. It retrieves relevant text passages and uses a language model to generate answers based on the uploaded document.

## 2. Objectives
- Extract text from PDF documents.
- Split extracted text into smaller chunks.
- Retrieve relevant information using vector similarity search.
- Generate answers using a language model.
- Provide a simple, interactive web interface.

## 3. Technologies Used
- Python
- Streamlit
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- FAISS
- Groq API
- python-dotenv

## 4. Project Workflow
1. Upload a PDF document.
2. Extract its readable text.
3. Split the text into overlapping chunks.
4. Convert chunks into numerical embeddings.
5. Store embeddings in a FAISS index.
6. Retrieve relevant chunks for the user's question.
7. Send the retrieved context to the Groq language model.
8. Display the generated answer.

## 5. Project Structure
- `app.py` – Streamlit user interface and question-answering flow.
- `document_loader.py` – Extracts text from PDF files.
- `vector_store.py` – Creates embeddings and searches for relevant chunks.
- `prompt.py` – Prompt-related functionality, if used by the application.
- `rag_pipeline.py` – RAG pipeline functionality, if used by the application.
- `requirements.txt` – Python dependencies.
- `tests/` – Project tests.
- `screenshots/` – Application screenshots.

## 6. Installation and Execution

### Prerequisites
- Python installed
- Internet connection for model downloads and Groq API requests
- A valid Groq API key

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure the API key
Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. Do not publish the real key.

### Run the application
```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## 7. Features
- PDF upload and text extraction
- Automatic text chunking
- Semantic retrieval using embeddings and FAISS
- AI-generated answers using Groq
- Expandable view of retrieved passages

## 8. Limitations
- Scanned PDFs may require OCR.
- Answers depend on the quality of extracted text and retrieved passages.
- The Groq API requires a valid key, internet access, and an available model.
- Summary requests may not cover the entire document when only selected chunks are retrieved.

## 9. Future Enhancements
- Support multiple PDFs.
- Add OCR for scanned documents.
- Display page references for answers.
- Improve retrieval and answer evaluation.
- Add conversation history and document management.

## 10. Conclusion
The PDF Document Assistant demonstrates how retrieval, vector search, and a language model can be combined to answer questions about PDF content through an interactive application.
