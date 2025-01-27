import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extracts text and metadata from a given PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Convert to UTF-8 encoding
                text += page_text.encode("utf-8").decode("utf-8") + "\n"
        metadata = reader.metadata
        return text, metadata
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return "", None

def chunk_text(text, chunk_size):
    """Chunks text into smaller pieces for vector storage."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_pdf_directory(directory_path, chunk_size=500):
    """Processes all PDF files in a directory and returns their text and metadata."""
    pdf_chunks = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text, metadata = extract_text_from_pdf(file_path)
            if text:
                chunks = chunk_text(text, chunk_size)
                pdf_chunks[filename] = {"chunks": chunks, "metadata": metadata}
    return pdf_chunks
