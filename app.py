
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from document_loader import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import create_vector_store, search_documents

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Assistant")
st.write("Upload a PDF and ask questions about its contents.")

# PDF upload
uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

# Process uploaded document
if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Reading and processing your PDF..."):
            try:
                os.makedirs("documents", exist_ok=True)

                pdf_path = os.path.join(
                    "documents",
                    os.path.basename(uploaded_file.name)
                )

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Extract text from PDF pages
                pages = extract_text_from_pdf(pdf_path)

                all_text = "\n".join(
                    page["text"]
                    for page in pages
                    if page.get("text", "").strip()
                )

                if not all_text.strip():
                    st.error(
                        "No readable text found in this PDF. "
                        "The PDF may be scanned."
                    )
                else:
                    # Split text into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=100
                    )

                    chunks = splitter.split_text(all_text)

                    if not chunks:
                        st.error("No text chunks could be created.")
                    else:
                        # Create FAISS vector store
                        model, index, stored_chunks = (
                            create_vector_store(chunks)
                        )

                        # Save processed data in session
                        st.session_state["search_data"] = (
                            model,
                            index,
                            stored_chunks
                        )

                        st.session_state["document_name"] = (
                            uploaded_file.name
                        )

                        st.session_state["processed"] = True

                        st.success(
                            f"Document processed successfully! "
                            f"Created {len(chunks)} text chunks."
                        )

            except Exception as e:
                st.error(f"Could not process the PDF: {e}")


# Question-answering section
if st.session_state.get("processed", False):
    st.subheader("💬 Ask a question")

    question = st.text_input(
        "Enter your question about the document",
        placeholder="e.g. Give me a summary of this PDF"
    )

    if st.button("Find Answer"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                model, index, chunks = (
                    st.session_state["search_data"]
                )

                # Retrieve relevant document chunks
                with st.spinner("Searching your document..."):
                    results = search_documents(
                        question,
                        model,
                        index,
                        chunks,
                        top_k=5
                    )

                if not results:
                    st.warning(
                        "I could not find this information "
                        "in the uploaded documents."
                    )

                else:
                    # Combine retrieved passages
                    context = "\n\n".join(results)

                    # Check API key
                    api_key = os.getenv("GROQ_API_KEY")

                    if not api_key:
                        st.error(
                            "Groq API key not found. "
                            "Check your .env file."
                        )

                    else:
                        client = Groq(api_key=api_key)

                        # Generate one grounded answer
                        with st.spinner("Generating AI answer..."):
                            response = (
                                client.chat.completions.create(
                                    model="openai/gpt-oss-120b",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": (
                                                "You are a document "
                                                "question-answering assistant. "
                                                "Answer using only the supplied "
                                                "context. Do not invent facts. "
                                                "If the answer is unavailable, "
                                                "say: 'I could not find this "
                                                "information in the uploaded "
                                                "documents.' For summary "
                                                "requests, summarize the "
                                                "provided context clearly "
                                                "using headings and bullet "
                                                "points where appropriate. "
                                                "Use simple English."
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": (
                                                f"Document: "
                                                f"{st.session_state.get('document_name', 'Uploaded PDF')}"
                                                f"\n\nContext:\n{context}"
                                                f"\n\nQuestion: {question}"
                                            )
                                        }
                                    ],
                                    temperature=0.2
                                )
                            )

                        answer = (
                            response.choices[0]
                            .message.content
                        )

                        st.subheader("📝 Answer")
                        st.markdown(answer)

                        with st.expander(
                            "View retrieved passages"
                        ):
                            for i, result in enumerate(
                                results, start=1
                            ):
                                st.markdown(
                                    f"**Passage {i}**"
                                )
                                st.write(result)

            except Exception as e:
                st.error(
                    f"Could not generate an answer: {e}"
                )

