import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
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
    client = OpenAI(api_key=openai_api_key)

    
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

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    # Add this in the sidebar section
with st.sidebar:
    # ... existing sidebar code (summary_option, use_advanced_model, etc.) ...
    # summary_option = st.radio(
    #     "Choose a summary method",
    #     (
    #         "Summarize the document in 100 words",
    #         "Summarize the document in 2 connecting paragraphs",
    #         "Summarize the document in 5 bullet points"
    #     )
    # )

    # use_advanced_model = st.checkbox("Use Advanced Model", value=False)
    # use_other_provider = st.checkbox("Use Other Provider", value=False)
    # if use_other_provider:
    provider_choice = st.radio("Choose Provider", ("GPT","Gemini", "Cohere"))
    if provider_choice == "Gemini":
        model = st.radio("Choose Gemini Model", ("gemini-2.0-flash-exp","gemini-2.5-pro"))
    elif provider_choice=="Cohere":
        model = st.radio("Choose Cohere Model", ("command-r7b-12-2024","command-a-reasoning-08-2025", "command-nightly"))
    else:
        model = st.radio("Choose a model", ("gpt-5-mini", "gpt-5-nano","gpt-4o"))
    # elif use_advanced_model:
    #     model = st.radio("Choose advanced model option", ("gpt-4o",))
    # else :
    #     # Only show basic models when neither advanced nor other provider is selected
    #     model = st.radio("Choose a model", ("gpt-5-mini", "gpt-5-nano","gpt-4o"))
    
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
