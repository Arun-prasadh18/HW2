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

