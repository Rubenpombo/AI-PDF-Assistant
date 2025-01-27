import streamlit as st
from pdf_processing import extract_text_from_pdf, chunk_text
from chromadb_setup import setup_chromadb, store_documents_in_chromadb, query_chromadb
import time
from logging_setup import setup_log
from llm_setup import query_llm

global logger

# Store query history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# File size limit (in bytes) - adjust as necessary
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def run_ui():
    # Set up full-width page
    st.set_page_config(layout="wide")
    # Set up logging
    logger = setup_log()

    # Initialize ChromaDB client, collection and vector_store in session state if not exists
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = setup_chromadb()

    # Initialize uploaded_files in session state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Create two columns: left for file upload and preview, center for querying
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.title("📄 PDF Uploader")

        # File Upload Section
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        # File Preview Section
        if uploaded_files:
            st.write("### File Previews")
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_size = uploaded_file.size
                
                # Expandable preview for each file
                with st.expander(f"{file_name} ({file_size/1024:.2f} KB)"):
                    try:
                        # Extract text preview
                        text, metadata = extract_text_from_pdf(uploaded_file)
                        
                        # Display first 500 characters
                        st.text(text[:500] + "..." if len(text) > 500 else text)
                        
                        # Display basic metadata
                        st.write("**Metadata:**")
                        st.json({k: v for k, v in metadata.items() if v is not None})
                    
                    except Exception as e:
                        st.error(f"Error previewing {file_name}: {e}")

        # Ingest Button
        if uploaded_files and st.button("Ingest Documents"):
            with st.spinner("Processing and storing documents..."):
                pdf_data = {}
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    text, metadata = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        chunks = chunk_text(text, chunk_size=500)
                        pdf_data[file_name] = {"chunks": chunks, "metadata": metadata}
                
                # Ingest PDF data into ChromaDB
                if pdf_data:
                    try:
                        store_documents_in_chromadb(st.session_state.vector_store, pdf_data)
                        st.success("Documents successfully stored!")
                        st.session_state.uploaded_files = uploaded_files
                    except Exception as e:
                        st.error(f"Ingestion error: {e}")

        # Clear Database Button                
        if st.button("Clear Database"):
            vector_store = setup_chromadb()
            logger.info("Database cleared successfully.")
            st.success("Database cleared successfully!")

    with right_col:
        st.title("🔍 Query PDF Documents")  
                             
        # Query Section
        query = st.text_input("🔍 Ask a question about your PDFs:")

        #Display query history
        if st.session_state.query_history:
            st.write("### Query History")
            for previous_query in st.session_state.query_history:
                if st.button(f"Re-run: {previous_query}"):
                    query = previous_query
                    st.text_input("🔍 Ask a question about your PDFs:", value=query)

        if query:
            with st.spinner('Processing your query...'):
                try:
                    results = query_chromadb(st.session_state.vector_store, query, n_results = 3)
                    logger.info(f"Results for query {query} from ChromaDB: {results}")
                    context_chunks = [doc.page_content for doc, score in results]
                    results_llm = query_llm(query, context_chunks, "es")

                    if results_llm:
                        st.write("### Query Results")
                        st.write(results_llm)  
                        
                        # Context sources
                        with st.expander("Context Sources"):
                            for doc, score in results:
                                    st.write(f"**Metadata:** {doc.metadata}")
                                    st.write(f"**Relevance Score:** {score}")

                        # Add query to history
                        if query not in st.session_state.query_history:
                            st.session_state.query_history.append(query)

                        # Export results button
                        export_data = f"Query: {query}\n\nLLM Response:\n{results_llm}\n\nContext Sources:\n"
                        for i, (doc, score) in enumerate(results):
                            export_data += f"Metadata {i+1}:\n{doc.metadata}\n\n"

                        st.download_button(
                            label="Export Results",
                            data=export_data,
                            file_name="query_results.txt",
                            mime="text/plain"
                        )

                    else:
                        st.write("No results found for your query.")

                except Exception as e:
                    st.error(f"Error during query: {e}")

if __name__ == "__main__":
    run_ui()
