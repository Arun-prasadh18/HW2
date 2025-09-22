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
# Create an OpenAI client if it doesn't exist in the session state.
if 'openai_client' not in st.session_state:
    try:
        api_key = st.secrets["API_KEY"]
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
    pdf_directory = "pdf_files" 
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

# --- Main App Logic ---

st.title("Chat With Your Documents")
st.info("Ask a question and the chatbot will answer based on the content of your uploaded PDF files.")


# Set up or retrieve the vector database from session state
if 'Lab4_vectorDB' not in st.session_state:
    st.session_state.Lab4_vectorDB = setup_vector_db()

collection = st.session_state.Lab4_vectorDB

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. RETRIEVE relevant context from ChromaDB
            openai_client = st.session_state.openai_client
            
            response = openai_client.embeddings.create(
                input=prompt,
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3 
            )
            
            # 2. AUGMENT the prompt with the retrieved context
            if results['documents']:
                context_documents = results['documents'][0]
                context_ids = results['ids'][0]
                
                context_str = "\n\n".join(f"Content from {context_ids[i]}:\n{doc}" for i, doc in enumerate(context_documents))

                system_prompt = f"""
                You are a helpful assistant who answers questions based on the provided context from documents.
                Your task is to use the following pieces of context to answer the user's question.
                If the answer is found within the context, you must clearly state that your answer is based on the provided documents and cite the source document names (e.g., 'According to document X.pdf...').
                If the context is not relevant or does not contain the answer, you should state that the documents do not provide an answer and then proceed to answer the question based on your general knowledge.

                Context from documents:
                ---
                {context_str}
                ---
                """
            else:
                 system_prompt = """
                 You are a helpful assistant. The user is asking a question, but no relevant documents were found to provide context.
                 Please inform the user that their uploaded documents do not contain relevant information and then answer their question based on your general knowledge.
                 """
            
            # 3. GENERATE a response using the LLM
            messages_for_api = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Using gpt-3.5-turbo as it is effective and fast.
            response_from_openai = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_for_api
            )
            
            full_response = response_from_openai.choices[0].message.content
            st.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

