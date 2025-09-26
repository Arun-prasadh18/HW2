import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# Fix for working with ChromaDB and Streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# --- OpenAI Client Initialization ---
if 'openai_client' not in st.session_state:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.session_state.openai_client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()

def add_to_collection(collection, text, filename, metadata={"source": "pdf"}):
    """
    Adds a document and its embedding to the ChromaDB collection.
    """
    openai_client = st.session_state.openai_client
    
    # Create an embedding for the text
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    # Add the document, metadata, and embedding to the collection
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding],
        metadatas=[metadata]
    )

def setup_vector_db():
    """
    Sets up the ChromaDB vector database by reading PDFs from a directory,
    creating embeddings, and storing them in a collection.
    """
    st.write("Setting up the vector database for the first time...")
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_lab")
    collection = chroma_client.get_or_create_collection("Lab4Collection")

    # Path to the directory containing PDF files
    pdf_directory = "PDF files" 
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        st.warning(f"Created directory '{pdf_directory}'. Please upload your PDF files there.")
        return collection

    # Process each PDF in the directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            # Check if the document is already in the collection
            if not collection.get(ids=[filename])['ids']:
                st.write(f"Processing {filename}...")
                try:
                    # Read PDF and extract text
                    pdf_path = os.path.join(pdf_directory, filename)
                    reader = PdfReader(pdf_path)
                    text = "".join(page.extract_text() for page in reader.pages)
                    
                    if text:
                        # Add the extracted text to the collection
                        add_to_collection(collection, text, filename)
                    else:
                        st.warning(f"No text extracted from {filename}.")
                except Exception as e:
                    st.error(f"Could not process {filename}: {e}")
            else:
                st.write(f"'{filename}' already exists in the collection. Skipping.")

    st.success("Vector database setup is complete!")
    return collection

def get_relevant_course_info(query, collection, n_results=3):
    """
    Takes a query as input and returns relevant course information from vector search.
    
    Args:
        query: The user's question/query string
        collection: The ChromaDB collection to search
        n_results: Number of results to retrieve
    
    Returns:
        A dictionary containing:
        - 'context': The concatenated relevant text from documents
        - 'sources': List of source document names
        - 'found': Boolean indicating if relevant docs were found
    """
    openai_client = st.session_state.openai_client
    
    # Create embedding for the query
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Perform vector search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Process and return results
    if results['documents'] and results['documents'][0]:
        context_documents = results['documents'][0]
        context_ids = results['ids'][0]
        
        # Build context string with source attribution
        context_parts = []
        for i, doc in enumerate(context_documents):
            context_parts.append(f"From {context_ids[i]}:\n{doc}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        return {
            'context': context_str,
            'sources': context_ids,
            'found': True
        }
    else:
        return {
            'context': "",
            'sources': [],
            'found': False
        }

def generate_response_with_memory(query, relevant_info, conversation_history):
    """
    Generates a response using the LLM with relevant course info and conversation memory.
    
    Args:
        query: The user's current question
        relevant_info: Dictionary with context from vector search
        conversation_history: List of previous messages for short-term memory
    
    Returns:
        The assistant's response string
    """
    openai_client = st.session_state.openai_client
    
    # Build the system prompt based on whether relevant docs were found
    if relevant_info['found']:
        system_prompt = f"""You are a helpful assistant with access to course documents. 
You have short-term memory of the conversation and should maintain context across messages.

IMPORTANT: When answering based on the provided documents, you MUST:
1. Clearly state that your answer comes from the provided documents
2. Cite the specific source document names (e.g., 'According to document X.pdf...')

Here is the relevant information from the course documents:

{relevant_info['context']}

Use this information to answer the user's question. If the information doesn't fully answer the question, you can supplement with your general knowledge but clearly distinguish between what comes from the documents and what is general knowledge."""
    else:
        system_prompt = """You are a helpful assistant with short-term memory of the conversation.
No relevant course documents were found for this query, but you should still help the user based on your general knowledge.
Maintain context from previous messages in the conversation."""
    
    # Build messages list including conversation history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (excluding system messages)
    for msg in conversation_history:
        if msg["role"] != "system":
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    # Generate response
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# --- Main App Logic ---
st.title("📚 Short-Term Memory Chatbot with Course Documents")
st.info("This chatbot remembers your conversation and can answer questions based on uploaded PDF files.")

# Set up or retrieve the vector database
if 'Lab4_vectorDB' not in st.session_state:
    st.session_state.Lab4_vectorDB = setup_vector_db()

collection = st.session_state.Lab4_vectorDB

# Initialize chat history for short-term memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents or anything else:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            # Step 1: Get relevant course information using vector search
            relevant_info = get_relevant_course_info(prompt, collection, n_results=3)
            
            # Step 2: Generate response with LLM using both vector search results and conversation memory
            response = generate_response_with_memory(
                query=prompt,
                relevant_info=relevant_info,
                conversation_history=st.session_state.messages[:-1]  # Exclude the just-added user message
            )
            
            st.markdown(response)
    
    # Add assistant response to chat history for memory
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot features:
    - **Short-term memory**: Remembers your conversation
    - **Document search**: Finds relevant info from PDFs
    - **Intelligent responses**: Combines document knowledge with general AI knowledge
    """)
    
    st.header("📁 Document Status")
    if collection:
        doc_count = len(collection.get()['ids'])
        st.success(f"✅ {doc_count} documents loaded")
    else:
        st.warning("⚠️ No documents loaded yet")
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()