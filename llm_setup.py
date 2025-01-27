from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from logging_setup import setup_log

# Initialize logging
logger = setup_log()

def initialize_llm():
    """Initialize the LLM with desired configuration."""
    return ChatOllama(
        model="llama3",
        temperature=0,
        top_p=0.9,
        max_tokens=500,
        stop=["\n", "User:"]
    )

def query_llm(query, context_chunks, language="en"):
    """
    Use the LLM to process the user query given the context chunks.

    Args:
        query (str): The user's query.
        context_chunks (list of str): Relevant chunks retrieved from ChromaDB.

    Returns:
        str: The LLM's response.
    """
    llm = initialize_llm()

    # Combine context chunks into a single context string
    context = "\n\n".join(context_chunks)

    # Set the language instruction based on the user's language
    language_instruction = {
        "en": "You are a helpful assistant for answering questions based on the provided context. Please answer in English.",
        "es": "Eres un asistente que responde preguntas basandote en el contexto proporcionado. Por favor, responde en español.",
        "fr": "Vous êtes un assistant utile pour répondre aux questions en fonction du contexte fourni. Veuillez répondre en français.", 
    }.get(language, "You are a helpful assistant for answering questions based on the provided context. Please answer in English.")  # Default to English if unsupported language

    # Create the input messages for the LLM
    messages = [
        SystemMessage(content = f"{language_instruction}"),
        HumanMessage(content = f"Context:\n{context}\n\nQuestion: {query}")
    ]

    try:
        # Get the response from the LLM
        response = llm.invoke(messages)
        return response.content if isinstance(response, AIMessage) else "Error: Unexpected response type."
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return "Sorry, I couldn't process your query."
