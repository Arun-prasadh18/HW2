# import streamlit as st
# from openai import OpenAI
# import google.generativeai as genai
# import cohere
# from bs4 import BeautifulSoup
# import requests
# # Show title and description.
# st.title("💬 Chatbot")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )
# cohere_api_key = st.secrets["Cohere_API_Key"]

# openai_api_key  = st.secrets["API_KEY"]
# Gemini_api_key=st.secrets["Gemini_API_Key"]
# GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
# def get_conversation_memory(messages, memory_option):
#     if memory_option == "Buffer of 6 questions":
#         # Return last 12 messages (6 user + 6 assistant pairs)
#         return messages[-12:]
    
#     elif memory_option == "Conversation summary":
#         if len(messages) <= 4:  # Keep first few messages as-is
#             return messages
#         # Keep first message and create summary of middle messages
#         first_msg = messages[0]
#         recent_msgs = messages[-4:]  # Keep last 4 messages
        
#         # Create summary of middle messages
#         middle_msgs = messages[1:-4]
#         if middle_msgs:
#             summary_content = "Previous conversation summary:\n"
#             for i in range(0, len(middle_msgs), 2):
#                 if i+1 < len(middle_msgs):
#                     user_msg = middle_msgs[i]["content"][:100]  # Truncate long messages
#                     assistant_msg = middle_msgs[i+1]["content"][:100]
#                     summary_content += f"User asked: {user_msg}...\nAssistant: {assistant_msg}...\n"
            
#             summary_msg = {"role": "system", "content": summary_content}
#             return [first_msg, summary_msg] + recent_msgs
#         else:
#             return messages
    
#     elif memory_option == "Buffer of 2,000 tokens":
#         # Approximate tokens by character count (rough estimate: 1 token ≈ 4 characters)
#         char_limit = 8000  # Roughly 2000 tokens
#         truncated = []
#         total_chars = 0
        
#         # Start from most recent messages
#         for msg in reversed(messages):
#             msg_chars = len(msg["content"])
#             if total_chars + msg_chars > char_limit and truncated:
#                 break
#             truncated.insert(0, msg)
#             total_chars += msg_chars
        
#         return truncated
    
#     else:
#         return messages
# from bs4 import BeautifulSoup
# import requests

# with st.sidebar:
    
#     provider_choice = st.radio("Choose Provider", ("GPT","Gemini", "Cohere","Groq"))
#     if provider_choice == "Gemini":
#         model = st.radio("Choose Gemini Model", ("gemini-2.0-flash-exp","gemini-2.5-pro"))
#     elif provider_choice=="Cohere":
#         model = st.radio("Choose Cohere Model", ("command-r7b-12-2024","command-a-reasoning-08-2025", "command-nightly"))
#     elif provider_choice=="Groq":
#         model = st.radio("Choose llama Model", ("llama-3.1-8b-instant","llama-3.3-70b-versatile"))
#     else:
#         model = st.radio("Choose a model", ("gpt-5-mini", "gpt-5-nano","gpt-4o"))
    
    
#     # Conversation memory options
#     st.divider()  # Optional: visual separator
#     memory_option = st.selectbox(
#         "Choose conversation memory type",
#         [
#             "Buffer of 6 questions",
#             "Conversation summary", 
#             "Buffer of 2,000 tokens"
#         ]
#     )

# # Ask user for their OpenAI API key via `st.text_input`.
# # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.secrets["API_KEY"]
# #st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:

#     # Create an OpenAI client.
#     # client = OpenAI(api_key=openai_api_key)
#     if openai_api_key:
#         openai_client = OpenAI(api_key=openai_api_key)
#     if Gemini_api_key:
#         genai.configure(api_key=Gemini_api_key)
#     if cohere_api_key:
#         # cohere_client = cohere.Client(api_key=cohere_api_key)
#         cohere_client = cohere.ClientV2(api_key=cohere_api_key)  # Use ClientV2

    
#     # Create a session state variable to store the chat messages. This ensures that the
#     # messages persist across reruns.
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display the existing chat messages via `st.chat_message`.
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Create a chat input field to allow the user to enter a message. This will display
#     # automatically at the bottom of the page.
#     if prompt := st.chat_input("What is up?"):

#         # Store and display the current prompt.
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Process URLs if provided
#         url_context = ""
#         if url_input or second_url_input:
#             url_context = process_urls(url_input, second_url_input)

#         # Apply conversation memory before sending to API
#         messages_to_send = get_conversation_memory(st.session_state.messages, memory_option)

#         # Add URL context to the current prompt if URLs provided
#         if url_context:
#             current_prompt = f"{prompt}\n\nAdditional context from URLs:{url_context}"
#             messages_to_send[-1]["content"] = current_prompt

#         # Generate response based on provider
#         with st.chat_message("assistant"):
#             if provider_choice == "GPT":
#                 stream = openai_client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": m["role"], "content": m["content"]} for m in messages_to_send],
#                     stream=True,
#                 )
#                 st.write("Selected model:", model)
#                 response = st.write_stream(stream)
                
#             elif provider_choice == "Gemini":
#                 try:
#                     gemini_model = genai.GenerativeModel(model)
#                     # Convert messages to Gemini format
#                     conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
#                     gemini_response = gemini_model.generate_content(conversation_text)
#                     response = gemini_response.text
#                     st.write("Selected model:", model)
#                     st.write(response)
#                 except Exception as e:
#                     response = f"Gemini API Error: {str(e)}"
#                     st.write(response)
#             elif provider_choice=="Groq":
#                 try:
#                     groq_client = OpenAI(
#                         base_url="https://api.groq.com/openai/v1",
#                         api_key=st.secrets["GROQ_API_KEY"]
#                     )
#                     stream = groq_client.chat.completions.create(
#                         model=model,
#                         messages=[{"role": m["role"], "content": m["content"]} for m in messages_to_send],
#                         stream=True,
#                     )
#                     st.write("Selected model:", model)
#                     response = st.write_stream(stream)
#                 except Exception as e:
#                     response = f"Groq API Error: {str(e)}"
#                     st.write(response)

            
#         st.session_state.messages.append({"role": "assistant", "content": response})

#     # Add this in the sidebar section
#     # Debug info (remove in production)
# if st.sidebar.checkbox("Show Debug Info"):
#     st.sidebar.write(f"Total messages: {len(st.session_state.messages)}")
#     st.sidebar.write(f"Last message: {st.session_state.messages[-1] if st.session_state.messages else 'None'}")
# # Prevent conversation loss during UI changes
# if "last_provider" not in st.session_state:
#     st.session_state.last_provider = provider_choice
    
# if st.session_state.last_provider != provider_choice:
#     st.session_state.last_provider = provider_choice
#     st.rerun()


# import streamlit as st
# from openai import OpenAI
# import os
# from PyPDF2 import PdfReader

# # Fix for working with ChromaDB and Streamlit
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import chromadb

# # --- OpenAI Client Initialization ---
# # Create an OpenAI client if it doesn't exist in the session state.
# if 'openai_client' not in st.session_state:
#     try:
#         api_key = st.secrets["OPENAI_API_KEY"]
#         st.session_state.openai_client = OpenAI(api_key=api_key)
#     except Exception as e:
#         st.error(f"Failed to initialize OpenAI client: {e}")
#         st.stop()

# def add_to_collection(collection, text, filename, metadata={"source": "pdf"}):
#     """
#     Adds a document and its embedding to the ChromaDB collection.
#     """
#     openai_client = st.session_state.openai_client
    
#     # Create an embedding for the text
#     response = openai_client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )
#     embedding = response.data[0].embedding

#     # Add the document, metadata, and embedding to the collection
#     collection.add(
#         documents=[text],
#         ids=[filename],
#         embeddings=[embedding],
#         metadatas=[metadata]
#     )

# def setup_vector_db():
#     """
#     Sets up the ChromaDB vector database by reading PDFs from a directory,
#     creating embeddings, and storing them in a collection.
#     """
#     st.write("Setting up the vector database for the first time...")
#     chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_lab")
#     collection = chroma_client.get_or_create_collection("Lab4Collection")

#     # Path to the directory containing PDF files
#     pdf_directory = "pdf_files" 
#     if not os.path.exists(pdf_directory):
#         os.makedirs(pdf_directory)
#         st.warning(f"Created directory '{pdf_directory}'. Please upload your PDF files there.")
#         return collection

#     # Process each PDF in the directory
#     for filename in os.listdir(pdf_directory):
#         if filename.endswith(".pdf"):
#             # Check if the document is already in the collection
#             if not collection.get(ids=[filename])['ids']:
#                 st.write(f"Processing {filename}...")
#                 try:
#                     # Read PDF and extract text
#                     pdf_path = os.path.join(pdf_directory, filename)
#                     reader = PdfReader(pdf_path)
#                     text = "".join(page.extract_text() for page in reader.pages)
                    
#                     if text:
#                         # Add the extracted text to the collection
#                         add_to_collection(collection, text, filename)
#                     else:
#                         st.warning(f"No text extracted from {filename}.")
#                 except Exception as e:
#                     st.error(f"Could not process {filename}: {e}")
#             else:
#                 st.write(f"'{filename}' already exists in the collection. Skipping.")

#     st.success("Vector database setup is complete!")
#     return collection

# # --- Main App Logic ---

# st.title("Chat With Your Documents")
# st.info("Ask a question and the chatbot will answer based on the content of your uploaded PDF files.")


# # Set up or retrieve the vector database from session state
# if 'Lab4_vectorDB' not in st.session_state:
#     st.session_state.Lab4_vectorDB = setup_vector_db()

# collection = st.session_state.Lab4_vectorDB

# # --- Chat Interface ---

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("Ask a question about your documents:"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             # 1. RETRIEVE relevant context from ChromaDB
#             openai_client = st.session_state.openai_client
            
#             response = openai_client.embeddings.create(
#                 input=prompt,
#                 model="text-embedding-3-small"
#             )
#             query_embedding = response.data[0].embedding

#             results = collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=3 
#             )
            
#             # 2. AUGMENT the prompt with the retrieved context
#             if results['documents']:
#                 context_documents = results['documents'][0]
#                 context_ids = results['ids'][0]
                
#                 context_str = "\n\n".join(f"Content from {context_ids[i]}:\n{doc}" for i, doc in enumerate(context_documents))

#                 system_prompt = f"""
#                 You are a helpful assistant who answers questions based on the provided context from documents.
#                 Your task is to use the following pieces of context to answer the user's question.
#                 If the answer is found within the context, you must clearly state that your answer is based on the provided documents and cite the source document names (e.g., 'According to document X.pdf...').
#                 If the context is not relevant or does not contain the answer, you should state that the documents do not provide an answer and then proceed to answer the question based on your general knowledge.

#                 Context from documents:
#                 ---
#                 {context_str}
#                 ---
#                 """
#             else:
#                  system_prompt = """
#                  You are a helpful assistant. The user is asking a question, but no relevant documents were found to provide context.
#                  Please inform the user that their uploaded documents do not contain relevant information and then answer their question based on your general knowledge.
#                  """
            
#             # 3. GENERATE a response using the LLM
#             messages_for_api = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt}
#             ]
            
#             # Using gpt-3.5-turbo as it is effective and fast.
#             response_from_openai = openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages_for_api
#             )
            
#             full_response = response_from_openai.choices[0].message.content
#             st.markdown(full_response)

#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

import streamlit as st
import os
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import chromadb

# --- Initial Setup and Configuration ---

# Fix for ChromaDB compatibility with Streamlit's environment
# This is a common workaround needed in modern Streamlit apps using SQLite.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

    if 'gemini_client' not in st.session_state:
        gemini_api_key = st.secrets["Gemini_API_Key"]
        genai.configure(api_key=gemini_api_key)
        st.session_state.gemini_client = genai.GenerativeModel('gemini-1.5-pro-latest')

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
    
    provider_choice = st.radio(
        "Choose LLM Provider:",
        ("OpenAI (GPT-4o)", "Google (Gemini 1.5 Pro)", "Groq (Llama 3.1 8B)"),
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

    html_directory = "/workspaces/HW2/HTML files"
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

            # 2. AUGMENT the prompt with the retrieved context
            if context_str:
                system_prompt = f"""
                You are a helpful assistant who answers questions based on the provided document context and conversation history.
                Your task is to use the following pieces of context to answer the user's question.
                If the answer is found within the context, you must clearly state that your answer is based on the provided documents and cite the source document names (e.g., 'According to document X.html...').
                If the context does not contain the answer, state that the documents do not provide an answer and then answer based on your general knowledge.
                
                Context from documents:
                ---
                {context_str}
                ---
                """
            else:
                system_prompt = """
                You are a helpful assistant. No relevant documents were found to provide context for the user's question.
                Please inform the user that their uploaded documents do not contain relevant information, and then answer their question based on your general knowledge.
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
                
                elif provider_choice == "Google (Gemini 1.5 Pro)":
                    # For Gemini, we combine the system prompt and the latest user message.
                    # The chat history is handled by the model object itself.
                    gemini_history = []
                    for msg in conversation_buffer[:-1]: # All but the last message
                        role = "model" if msg["role"] == "assistant" else "user"
                        gemini_history.append({'role': role, 'parts': [msg['content']]})
                    
                    chat_session = st.session_state.gemini_client.start_chat(history=gemini_history)
                    
                    final_prompt = f"{system_prompt}\n\nUser Question: {prompt}"
                    
                    gemini_response = chat_session.send_message(final_prompt)
                    full_response = gemini_response.text
                    st.markdown(full_response)

            except Exception as e:
                full_response = f"An error occurred with the selected LLM provider: {e}"
                st.error(full_response)
    
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})