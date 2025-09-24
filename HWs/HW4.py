
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb

# --- Initial Setup and Configuration ---


# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with HTML Docs",
    page_icon="🤖",
    layout="wide"
)

st.title("💬 Chat with Your HTML Documents")
st.info("Ask a question, and the chatbot will answer based on the content of your HTML files, with conversational memory.")

# --- API Client Initialization ---

# Initialize API clients and store them in session state to avoid re-creating them on each rerun.
try:
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # MODIFIED: Removed the Gemini client initialization as it's no longer used.

    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.secrets["GROQ_API_KEY"]
        )
except Exception as e:
    st.error(f"Failed to initialize API clients. Please check your API keys in Streamlit secrets. Error: {e}")
    st.stop()


# --- Sidebar for LLM Selection ---
with st.sidebar:
    st.header("Configuration")
    st.write("Choose the Language Model to power the chat.")
    
    # MODIFIED: Replaced the Google Gemini option with another Groq model (Qwen).
    provider_choice = st.radio(
        "Choose LLM Provider:",
        ("OpenAI (GPT-4o)", "Groq (Llama 3.1 8B)", "Groq (Qwen3 32B)"),
        key="llm_provider"
    )
    st.divider()
    st.info("The vector database is created once from the files in the `html_files` directory.")


# --- Vector Database (ChromaDB) Functions ---

def add_chunks_to_collection(collection, text_chunks, filename):
    """Adds document chunks and their embeddings to the ChromaDB collection."""
    openai_client = st.session_state.openai_client
    
    # Generate unique IDs for each chunk
    ids = [f"{filename}_part_{i+1}" for i in range(len(text_chunks))]
    
    # Create embeddings for the text chunks
    try:
        response = openai_client.embeddings.create(
            input=text_chunks,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]

        # Add the documents, metadata, and embeddings to the collection
        collection.add(
            documents=text_chunks,
            ids=ids,
            embeddings=embeddings,
            metadatas=[{"source": filename} for _ in ids]
        )
        return True
    except Exception as e:
        st.error(f"Failed to create embeddings for {filename}: {e}")
        return False

def setup_vector_db():
    """
    Sets up the ChromaDB vector database. It reads HTML files, chunks them,
    creates embeddings, and stores them in a persistent collection.
    This process only runs if the database directory doesn't exist.
    """
    db_path = "./ChromaDB_HTML_Docs"
    if os.path.exists(db_path):
        st.sidebar.success("VectorDB already exists. Loading...")
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection("HTML_RAG_Collection")
        return collection

    st.sidebar.write("Setting up the vector database for the first time...")
    progress_bar = st.sidebar.progress(0)
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection("HTML_RAG_Collection")

    html_directory = "HTML files"
    if not os.path.exists(html_directory):
        os.makedirs(html_directory)
        st.warning(f"Created directory '{html_directory}'. Please add your HTML files there and refresh.")
        return collection

    html_files = [f for f in os.listdir(html_directory) if f.endswith(".html")]
    if not html_files:
        st.warning(f"No HTML files found in the '{html_directory}' directory.")
        return collection

    for i, filename in enumerate(html_files):
        st.sidebar.write(f"Processing {filename}...")
        try:
            filepath = os.path.join(html_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text(separator=" ", strip=True)

            if text:
                # --- CHUNKING METHOD ---
                # EXPLANATION: We are using a fixed-number chunking strategy. The goal is to
                # split each document into exactly two smaller "mini-documents".
                # To do this, we find the midpoint of the text by character length and split it.
                # WHY THIS METHOD: This approach is simple, fast, and guarantees that each
                # source document is represented by a consistent number of chunks (two).
                # This is useful for balancing the data representation in the vector store
                # without the complexity of semantic or recursive chunking, which might be
                # overkill for this specific requirement.
                midpoint = len(text) // 2
                chunk1 = text[:midpoint]
                chunk2 = text[midpoint:]
                text_chunks = [chunk1, chunk2]
                
                # Check if the first chunk is already in the collection to avoid reprocessing
                if not collection.get(ids=[f"{filename}_part_1"])['ids']:
                    add_chunks_to_collection(collection, text_chunks, filename)
                else:
                    st.sidebar.write(f"'{filename}' already in DB. Skipping.")
            else:
                st.warning(f"No text extracted from {filename}.")

        except Exception as e:
            st.error(f"Could not process {filename}: {e}")
        
        progress_bar.progress((i + 1) / len(html_files))

    st.sidebar.success("Vector database setup is complete!")
    return collection

# --- Conversational Memory Function ---

def get_conversation_buffer(messages):
    """
    Retrieves the last 5 pairs of user questions and assistant answers
    to maintain conversational context.
    """
    # Each turn consists of a user message and an assistant message (2 items).
    # We want the last 5 turns, so we take the last 10 messages.
    return messages[-10:]

# --- Main App Logic ---

# Set up or retrieve the vector database from session state
if 'vectorDB_collection' not in st.session_state:
    st.session_state.vectorDB_collection = setup_vector_db()
collection = st.session_state.vectorDB_collection

# Display the number of items (chunks) in the collection
try:
    item_count = collection.count()
    st.sidebar.metric("Chunks in DB", item_count)
except Exception as e:
    st.sidebar.error(f"Could not count items in DB: {e}")

# Initialize chat history in session state
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
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... Retrieving context and generating answer..."):
            # 1. RETRIEVE relevant context from ChromaDB
            openai_client = st.session_state.openai_client
            
            try:
                response = openai_client.embeddings.create(
                    input=prompt, model="text-embedding-3-small"
                )
                query_embedding = response.data[0].embedding

                results = collection.query(
                    query_embeddings=[query_embedding], n_results=3
                )
                
                context_documents = results['documents'][0] if results['documents'] else []
                context_ids = results['ids'][0] if results['ids'] else []
                context_str = "\n\n".join(f"Content from {context_ids[i]}:\n{doc}" for i, doc in enumerate(context_documents))

            except Exception as e:
                st.error(f"Failed to query the vector database: {e}")
                context_str = "" # Ensure context_str is defined

            # # 2. AUGMENT the prompt with the retrieved context
            # if context_str:
            #     system_prompt = f"""
            #     You are a helpful assistant who answers questions based on the provided document context and conversation history.
            #     Your task is to use the following pieces of context to answer the user's question.
            #     If the answer is found within the context, you must clearly state that your answer is based on the provided documents and cite the source document names (e.g., 'According to document X.html...').
            #     If the context does not contain the answer, state that the documents do not provide an answer and then answer based on your general knowledge.
                
            #     Context from documents:
            #     ---
            #     {context_str}
            #     ---
            #     """
            # else:
            #     system_prompt = """
            #     You are a helpful assistant. No relevant documents were found to provide context for the user's question.
            #     Please inform the user that their uploaded documents do not contain relevant information, and then answer their question based on your general knowledge.
            #     """
            # 2. AUGMENT the prompt with the retrieved context
            if context_str:
                system_prompt = f"""
                You are a helpful assistant who answers questions based on the provided document context and conversation history.
                Your task is to use the following pieces of context to answer the user's question.
                If the answer is found within the context, you must clearly state that your answer is based on the provided documents and cite the source document names (e.g., 'According to document X.html...').
                If the context does not contain the answer, state that the documents do not provide an answer and then answer based on your general knowledge.

                **IMPORTANT: Do not output any reasoning, thinking steps, or XML tags like <think>. Provide only the final, direct answer to the user.**
                
                Context from documents:
                ---
                {context_str}
                ---
                """
            else:
                system_prompt = """
                You are a helpful assistant. No relevant documents were found to provide context for the user's question.
                Please inform the user that their uploaded documents do not contain relevant information, and then answer their question based on your general knowledge.

                **IMPORTANT: Do not output any reasoning, thinking steps, or XML tags like <think></think>. Provide only the final, direct answer to the user.**
                """
            # 3. Get conversational memory
            conversation_buffer = get_conversation_buffer(st.session_state.messages)

            # 4. GENERATE a response using the selected LLM
            full_response = ""
            try:
                if provider_choice == "OpenAI (GPT-4o)":
                    messages_for_api = [
                        {"role": "system", "content": system_prompt},
                        *conversation_buffer  # Unpack the list of dicts
                    ]
                    stream = st.session_state.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages_for_api,
                        stream=True,
                    )
                    full_response = st.write_stream(stream)

                elif provider_choice == "Groq (Llama 3.1 8B)":
                    messages_for_api = [
                        {"role": "system", "content": system_prompt},
                        *conversation_buffer
                    ]
                    stream = st.session_state.groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages_for_api,
                        stream=True,
                    )
                    full_response = st.write_stream(stream)
                
                # MODIFIED: Replaced the Gemini logic with another Groq model call.
                elif provider_choice == "Groq (Qwen3 32B)":
                    messages_for_api = [
                        {"role": "system", "content": system_prompt},
                        *conversation_buffer
                    ]
                    # New code for Qwen
                    stream = st.session_state.groq_client.chat.completions.create(
                        model="qwen/qwen3-32b",
                        messages=messages_for_api,
                        stream=True,
                    )
                    # Manual streaming to intercept and clean the output in real-time
                    placeholder = st.empty()
                    full_response = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            # Clean the response on the fly by removing unwanted tags and leading whitespace
                            cleaned_response = full_response.replace("<think>", "").replace("</think>", "").lstrip()
                            placeholder.markdown(cleaned_response + "▌")
                    # Final render of the cleaned response without the cursor
                    cleaned_response = full_response.replace("<think>", "").replace("</think>", "").lstrip()
                    placeholder.markdown(cleaned_response)

            except Exception as e:
                full_response = f"An error occurred with the selected LLM provider: {e}"
                st.error(full_response)
    
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})