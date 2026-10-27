
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def create_vector_store(chunks):
    # Convert text into numerical vectors
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Create a searchable index
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks


def search_documents(query, model, index, chunks, top_k=3):
    # Convert the question into a vector
    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    # Find the closest text chunks
    distances, indices = index.search(
        query_vector, min(top_k, len(chunks))
    )

    results = []
    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])

    return results
