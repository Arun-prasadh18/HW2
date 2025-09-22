import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import cohere
from bs4 import BeautifulSoup
import requests
# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)
cohere_api_key = st.secrets["Cohere_API_Key"]

openai_api_key  = st.secrets["API_KEY"]
Gemini_api_key=st.secrets["Gemini_API_Key"]
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
def get_conversation_memory(messages, memory_option):
    if memory_option == "Buffer of 6 questions":
        # Return last 12 messages (6 user + 6 assistant pairs)
        return messages[-12:]
    
    elif memory_option == "Conversation summary":
        if len(messages) <= 4:  # Keep first few messages as-is
            return messages
        # Keep first message and create summary of middle messages
        first_msg = messages[0]
        recent_msgs = messages[-4:]  # Keep last 4 messages
        
        # Create summary of middle messages
        middle_msgs = messages[1:-4]
        if middle_msgs:
            summary_content = "Previous conversation summary:\n"
            for i in range(0, len(middle_msgs), 2):
                if i+1 < len(middle_msgs):
                    user_msg = middle_msgs[i]["content"][:100]  # Truncate long messages
                    assistant_msg = middle_msgs[i+1]["content"][:100]
                    summary_content += f"User asked: {user_msg}...\nAssistant: {assistant_msg}...\n"
            
            summary_msg = {"role": "system", "content": summary_content}
            return [first_msg, summary_msg] + recent_msgs
        else:
            return messages
    
    elif memory_option == "Buffer of 2,000 tokens":
        # Approximate tokens by character count (rough estimate: 1 token ≈ 4 characters)
        char_limit = 8000  # Roughly 2000 tokens
        truncated = []
        total_chars = 0
        
        # Start from most recent messages
        for msg in reversed(messages):
            msg_chars = len(msg["content"])
            if total_chars + msg_chars > char_limit and truncated:
                break
            truncated.insert(0, msg)
            total_chars += msg_chars
        
        return truncated
    
    else:
        return messages
from bs4 import BeautifulSoup
import requests
def process_urls(url1, url2=None):
    """Extract text content from URLs"""
    url_content = ""
    
    for url in [url1, url2]:
        if url:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                url_content += f"\n\nContent from {url}:\n{text[:2000]}..."  # Limit to 2000 chars
                
            except Exception as e:
                url_content += f"\n\nError processing {url}: {str(e)}"
    
    return url_content

with st.sidebar:
    
    provider_choice = st.radio("Choose Provider", ("GPT","Gemini", "Cohere","Groq"))
    if provider_choice == "Gemini":
        model = st.radio("Choose Gemini Model", ("gemini-2.0-flash-exp","gemini-2.5-pro"))
    elif provider_choice=="Cohere":
        model = st.radio("Choose Cohere Model", ("command-r7b-12-2024","command-a-reasoning-08-2025", "command-nightly"))
    elif provider_choice=="Groq":
        model = st.radio("Choose llama Model", ("llama-3.1-8b-instant","llama-3.3-70b-versatile"))
    else:
        model = st.radio("Choose a model", ("gpt-5-mini", "gpt-5-nano","gpt-4o"))
    
    # Add these new options at the end of sidebar before API key section
    st.divider()  # Optional: visual separator
    url_input = st.text_input(
    "Enter First URL to summarize",
    "",
    placeholder="Paste the website URL here...")
    # Additional URL input
    second_url_input = st.text_input(
        "Enter a second URL (optional)",
        "",
        placeholder="Paste the second website URL here..."
    )
    
    # Conversation memory options
    st.divider()  # Optional: visual separator
    memory_option = st.selectbox(
        "Choose conversation memory type",
        [
            "Buffer of 6 questions",
            "Conversation summary", 
            "Buffer of 2,000 tokens"
        ]
    )

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["API_KEY"]
#st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    # client = OpenAI(api_key=openai_api_key)
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    if Gemini_api_key:
        genai.configure(api_key=Gemini_api_key)
    if cohere_api_key:
        # cohere_client = cohere.Client(api_key=cohere_api_key)
        cohere_client = cohere.ClientV2(api_key=cohere_api_key)  # Use ClientV2

    
    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # # Generate a response using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # Apply conversation memory before sending to API
        # messages_to_send = get_conversation_memory(st.session_state.messages, memory_option)

        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": m["role"], "content": m["content"]}
        #         for m in messages_to_send
        #     ],
        #     stream=True,
        # )

        # # Stream the response to the chat using `st.write_stream`, then store it in 
        # # session state.
        # with st.chat_message("assistant"):
        #     response = st.write_stream(stream)
        # st.session_state.messages.append({"role": "assistant", "content": response})
        # Process URLs if provided
        url_context = ""
        if url_input or second_url_input:
            url_context = process_urls(url_input, second_url_input)

        # Apply conversation memory before sending to API
        messages_to_send = get_conversation_memory(st.session_state.messages, memory_option)

        # Add URL context to the current prompt if URLs provided
        if url_context:
            current_prompt = f"{prompt}\n\nAdditional context from URLs:{url_context}"
            messages_to_send[-1]["content"] = current_prompt

        # Generate response based on provider
        with st.chat_message("assistant"):
            if provider_choice == "GPT":
                stream = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages_to_send],
                    stream=True,
                )
                st.write("Selected model:", model)
                response = st.write_stream(stream)
                
            elif provider_choice == "Gemini":
                try:
                    gemini_model = genai.GenerativeModel(model)
                    # Convert messages to Gemini format
                    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
                    gemini_response = gemini_model.generate_content(conversation_text)
                    response = gemini_response.text
                    st.write("Selected model:", model)
                    st.write(response)
                except Exception as e:
                    response = f"Gemini API Error: {str(e)}"
                    st.write(response)
            elif provider_choice=="Groq":
                try:
                    groq_client = OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=st.secrets["GROQ_API_KEY"]
                    )
                    stream = groq_client.chat.completions.create(
                        model=model,
                        messages=[{"role": m["role"], "content": m["content"]} for m in messages_to_send],
                        stream=True,
                    )
                    st.write("Selected model:", model)
                    response = st.write_stream(stream)
                except Exception as e:
                    response = f"Groq API Error: {str(e)}"
                    st.write(response)

            # elif provider_choice=="Groq":
            #     try:

            # elif provider_choice == "Cohere":
            #     try:
            #         # Convert messages to Cohere v2 format
            #         cohere_messages = []
            #         for msg in messages_to_send:
            #             cohere_messages.append({
            #                 "role": "user" if msg["role"] == "user" else "assistant", 
            #                 "content": msg["content"]
            #             })
                    
            #         cohere_response = cohere_client.chat(
            #             model=model,
            #             messages=cohere_messages,
            #             max_tokens=1000
            #         )
                    
            #         # Handle different response content types
            #         if hasattr(cohere_response, 'message') and hasattr(cohere_response.message, 'content'):
            #             # Extract text from content array
            #             content_text = ""
            #             for content_item in cohere_response.message.content:
            #                 if hasattr(content_item, 'text'):
            #                     content_text += content_item.text
            #                 elif hasattr(content_item, 'content') and isinstance(content_item.content, str):
            #                     content_text += content_item.content
            #             response = content_text
            #         else:
            #             # Fallback - convert response to string
            #             response = str(cohere_response)
                        
            #         st.write("Selected model:", model)
            #         st.write(response)
                    
            #     except Exception as e:
            #         response = f"Cohere API Error: {str(e)}"
            #         st.write(response)
            #         # Debug: show response structure
            #         st.write("Debug - Response type:", type(cohere_response) if 'cohere_response' in locals() else "No response")

            # elif provider_choice == "Cohere":
            #     try:
            #         # Convert messages to conversation format for Cohere V2
            #         cohere_messages = []
            #         for msg in messages_to_send:
            #             cohere_messages.append({
            #                 "role": "user" if msg["role"] == "user" else "assistant",
            #                 "content": msg["content"]
            #             })
                    
            #         cohere_response = cohere_client.chat(
            #             model=model,
            #             messages=cohere_messages,
            #             max_tokens=1000
            #         )
            #         response = cohere_response.message.content[0].text
            #         st.write("Selected model:", model)
            #         st.write(response)
            #     except Exception as e:
            #         response = f"Cohere API Error: {str(e)}"
            #         st.write(response)
                
            # elif provider_choice == "Cohere":
            #     try:
            #         # Convert messages to conversation format
            #         conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
            #         cohere_response = cohere_client.generate(
            #             model=model,
            #             prompt=conversation_text,
            #             max_tokens=1000
            #         )
            #         response = cohere_response.generations[0].text
            #         st.write("Selected model:", model)
            #         st.write(response)
            #     except Exception as e:
            #         response = f"Cohere API Error: {str(e)}"
            #         st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add this in the sidebar section
    # Debug info (remove in production)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write(f"Total messages: {len(st.session_state.messages)}")
    st.sidebar.write(f"Last message: {st.session_state.messages[-1] if st.session_state.messages else 'None'}")
# Prevent conversation loss during UI changes
if "last_provider" not in st.session_state:
    st.session_state.last_provider = provider_choice
    
if st.session_state.last_provider != provider_choice:
    st.session_state.last_provider = provider_choice
    st.rerun()

# import streamlit as st
# from openai import OpenAI
# import os
# from PyPDF2 import PdfReader

# #fix for working with ChromaDB and streamlit
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# import chromadb
# # Initialize ChromaDB client
# chromaDB_path = "./ChromaDB_for_lab"

# chroma_client = chromadb.PersistentClient(chromaDB_path)
# collection = chroma_client.get_or_create_collection("Lab4Collection")

# #init openAI
# # Create an OpenAI client.
# if 'openai_client' not in st.session_state:
#     api_key = st.secrets["OPENAI_API_KEY"]
#     st.session_state.openai_client = OpenAI(api_key=api_key)
# # # Create an OpenAI client.
# # if 'openai_client' not in st.session_state:
# #     api_key = st.secrets["OPENAI_API_KEY"]
# #     st.session_state.openai_client = OpenAI(api_key=api_key)

# def add_to_collection(collection, text, filename):
    
#     # Create an embedding
#     openai_client = st.session_state.openai_client
#     response = openai_client.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )

#     # Get the embedding
#     embedding = response.data[0].embedding

#     # Add embedding and document to ChromaDB
#     collection.add(
#         documents=[text],
#         ids = [filename],
#         embeddings=[embedding]
#     )

# topic = st.sidebar.selectbox("Topic",
#     ("Text Mining", "GenAI"))

# openai_client = st.session_state.openai_client
# response = openai_client.embeddings.create(
#     input=topic,
#     model="text-embedding-3-small")

# # Get the embedding
# query_embedding = response.data[0].embedding

# #get the text relating to this question (this prompt)
# results = collection.query(
#     query_embeddings=[query_embedding],
#     n_results=3 # Number of closest documents to return
# )

# # Print the results with IDs using an index
# for i in range(len(results['documents'][0])):
#     doc = results['documents'][0][i]
#     doc_id = results['ids'][0][i]
#     st.write(f"The following file/syllabus might be helpful: {doc_id} ")

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
#         api_key = st.secrets["API_KEY"]
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
#     pdf_directory = "PDF files" 
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

# st.title("Document Search with VectorDB")

# # Set up or retrieve the vector database from session state
# if 'Lab4_vectorDB' not in st.session_state:
#     st.session_state.Lab4_vectorDB = setup_vector_db()

# collection = st.session_state.Lab4_vectorDB

# # --- Search Interface ---
# st.sidebar.header("Test the VectorDB")
# topic = st.sidebar.selectbox(
#     "Choose a topic to search for:",
#     ("Generative AI", "Text Mining", "Data Science Overview")
# )

# if st.sidebar.button("Search"):
#     with st.spinner(f"Searching for documents related to '{topic}'..."):
#         openai_client = st.session_state.openai_client
        
#         # Create an embedding for the search query
#         response = openai_client.embeddings.create(
#             input=topic,
#             model="text-embedding-3-small"
#         )
#         query_embedding = response.data[0].embedding

#         # Query the collection for the 3 most relevant documents
#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=3 
#         )

#         st.subheader("Search Results")
#         if results['documents']:
#             # Display the results
#             for i in range(len(results['documents'][0])):
#                 doc_id = results['ids'][0][i]
#                 st.write(f"**{i+1}.** The following file might be helpful: `{doc_id}`")
#         else:
#             st.write("No relevant documents found.")

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
#         api_key = st.secrets["API_KEY"]
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

