import streamlit as st
import PyPDF2
from openai import OpenAI
# pg=st.navigation([first_page,second_page])
st.set_page_config(page_title="Data Manager",page_icon=":material/edit:")
# with st.sidebar:
#     add_radio=st.radio("choose a option",("Summarize the document in 100 words", 
#                                           "Summarize the document in 2 connecting paragraphs",
#  "Summarize the document in 5 bullet points"))
#     model_radio=st.radio("choose a option",("GPT-MINI","GPT-NANO"))
#     use_advanced=st.checkbox("Use advanced model", value=False)
#     if use_advanced:
#         model_advanced_radio = st.radio(
#             "Choose advanced model option",
#             ("GPT-MINI",)  # Add more models if available
#         )
# Add at the very top of the main area (not sidebar)
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

    if use_advanced_model:
        model = st.radio(
            "Choose advanced model option",
            ("gpt-4o",)  # Add more models if available
        )
    else:
        model = st.radio(
            "Choose a model",
            ("gpt-5-mini", "gpt-5-nano")
        )

openai_api_key  = st.secrets["API_KEY"]

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    # uploaded_file = st.file_uploader(
    #     "Upload a document (.txt or .md)", type=("txt","pdf")
    # )

    # Ask the user for a question via `st.text_area`.
    question = summary_option
    document = url_content

    # if uploaded_file and question:
    #     file_extension = uploaded_file.name.split('.')[-1]
    #     if file_extension == 'txt':
    #         document = uploaded_file.read().decode()
    #     elif file_extension == 'pdf':
    #         document = read_pdf(uploaded_file)
    #     else:
    #         st.error("Unsupported file type.")
        # Process the uploaded file and question.
        # document = uploaded_file.read().decode()
        
    if document and model:
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question} \n\nPlease provide the summary in {output_language}.",
            }
        ]
    
        stream = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
        st.write(model)
        st.write(stream.choices[0].message.content)
