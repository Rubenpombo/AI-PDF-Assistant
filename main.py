from logging_setup import setup_log
from pdf_processing import process_pdf_directory
from chromadb_setup import setup_chromadb, store_documents_in_chromadb, query_chromadb, delete_collection


def main():
    """Main entry point for processing PDFs and storing them in ChromaDB."""
    # Set up logging
    logger = setup_log()

    # Configuration
    directory = "./pdf_documents"  # Replace with your PDF directory
    chunk_size = 500

    # Process PDF files
    logger.info("Processing PDF files...")
    uploaded_files = process_pdf_directory(directory, chunk_size)
    logger.info(f"Processing {len(uploaded_files)} uploaded PDF files...")


    # Setup ChromaDB client and collection
    logger.info("Setting up ChromaDB...")
    vector_store = setup_chromadb()

    # Clear the collection if it already exists
    # delete_collection(client, "pdf_documents")

    # Store documents in ChromaDB
    logger.info("Storing documents in ChromaDB...")
    store_documents_in_chromadb(vector_store, uploaded_files)
    logger.info("PDF documents have been ingested and stored in ChromaDB.")

    # Query ChromaDB with dynamic input and error handling
    try:
        query_text = input("Enter your query: ")  # Get query input from user

        results = query_chromadb(vector_store, query_text, n_results=1)

        for res, score in results:
            print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        logger.info(f"Query results for '{query_text}': {results}")

    except Exception as e:
        logger.error(f"Error during query: {e}")
    
if __name__ == "__main__":
    main()