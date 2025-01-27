import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


def setup_chromadb():
    """
    Initializes and configures a ChromaDB instance using the updated client architecture.
    For easier access to the underlying database, creating a client is better.
    """
    
    # Initialize the client with a persistent storage directory
    persistent_client = chromadb.PersistentClient(path="/chroma_storage")
    
    # Delete the existing collection if it exists
    delete_collection(persistent_client, "pdf_documents")

    # Initialize the Ollama embeddings model
    embeddings = OllamaEmbeddings(model="llama3")

    collection = persistent_client.get_or_create_collection(name = "pdf_documents")

    vector_store = Chroma(
    client = persistent_client,
    collection_name = "pdf_documents",
    embedding_function = embeddings,
    )


    return vector_store


def delete_collection(client, collection_name):
    """
    Deletes a collection from the ChromaDB client.
    """
    try:
        client.delete_collection(name = collection_name)
    except Exception as e:
        print(f"Could not delete collection: {e}")
    


def store_documents_in_chromadb(vector_store, pdf_data):
    """
    Stores PDF content and metadata in ChromaDB collection.
    ChromaDB automatically handles embeddings and indexing.
    """
    for doc_id, content in pdf_data.items():
        chunks = content["chunks"]
        metadata = content["metadata"]

        # Prepare documents and IDs for this file
        batch_docs = [
            Document(page_content=chunk, metadata={"filename": doc_id})
            for chunk in chunks
        ]
        batch_ids = [f"{doc_id}_{i}" for i in range(len(batch_docs))]

        # Add documents with proper IDs
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)



def query_chromadb(vector_store, query_text, n_results = 1):
    """
    Queries the ChromaDB collection with a given text query.
    Uses an LLM model to find similar documents based on embeddings.
    """
    try:
        results = vector_store.similarity_search_with_score(query_text, k = n_results)
        return results
    except Exception as e:
        raise e