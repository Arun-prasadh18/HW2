import streamlit as st
import PyPDF2
import google.generativeai as genai
from openai import OpenAI
import cohere
st.set_page_config(page_title="Data Manager",page_icon=":material/edit:")

url_input = st.text_input(
    "Enter a URL to summarize",
    "",
    placeholder="Paste the website URL here..."
)
output_language = st.selectbox(
    "Select output language",
    ["English", "French", "Spanish"]
)

import requests
from bs4 import BeautifulSoup
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

url_content = read_url_content(url_input) if url_input else None
with st.sidebar:
    summary_option = st.radio(
        "Choose a summary method",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        )
    )

    use_advanced_model = st.checkbox("Use Advanced Model", value=False)
    use_other_provider = st.checkbox("Use Other Provider", value=False)
    if use_other_provider:
        provider_choice = st.radio("Choose Provider", ("Gemini", "Cohere"))
        if provider_choice == "Gemini":
            model = st.radio("Choose Gemini Model", ("gemini-2.0-flash-exp","gemini-2.5-pro"))
        else:
            model = st.radio("Choose Cohere Model", ("command-r7b-12-2024","command-a-reasoning-08-2025", "command-nightly"))
    elif use_advanced_model:
        model = st.radio("Choose advanced model option", ("gpt-4o",))
    else:
        # Only show basic models when neither advanced nor other provider is selected
        model = st.radio("Choose a model", ("gpt-5-mini", "gpt-5-nano"))
    
cohere_api_key = st.secrets["Cohere_API_Key"]

openai_api_key  = st.secrets["API_KEY"]
Gemini_api_key=st.secrets["Gemini_API_Key"]

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


    # Ask the user for a question via `st.text_area`.
    question = summary_option
    document = url_content


    if document and model:
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question} \n\nPlease provide the summary in {output_language}.",
            }
        ]
        if use_other_provider:
            if model in ["gemini-2.0-flash-exp","gemini-2.5-pro"]:
                # Configure Gemini
                gemini_model = genai.GenerativeModel(model)
                
                # Extract content for Gemini
                prompt = messages[0]["content"]
                response = gemini_model.generate_content(prompt)
                
                st.write("Selected model:", model)
                st.write("Response:", response.text)
            
            elif model in ["command-r7b-12-2024", "command-a-reasoning-08-2025", "command-nightly"]:
                # Configure Cohere using Chat API v2
                try:
                    response = cohere_client.chat(
                        model=model,
                        messages=[{"role": "user", "content": messages[0]["content"]}],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    st.write("Selected model:", model)
                    
                    # Handle the nested content structure
                    if hasattr(response.message, 'content') and response.message.content:
                        # Find the text content (skip thinking content)
                        for content_item in response.message.content:
                            if hasattr(content_item, 'type') and content_item.type == 'text':
                                st.write("Response:", content_item.text)
                                break
                    else:
                        st.error("No text content found in response")
                    
                except Exception as e:
                    st.error(f"Error calling Cohere API: {str(e)}")

        else:
            # OpenAI models
            if openai_api_key:
                stream = openai_client.chat.completions.create(
                            model=model,
                            messages=messages
                        )
                st.write("Selected model:", model)
                st.write("Response:", stream.choices[0].message.content)
            else:
                st.error("OpenAI API key not found. Please add it to your secrets.")
