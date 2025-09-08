import streamlit as st
import PyPDF2
import google.generativeai as genai
from openai import OpenAI
import cohere
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
    # if use_advanced_model:
    #     model = st.radio(
    #         "Choose advanced model option",
    #         ("gpt-4o",)  # Add more models if available
    #     )
    # else:
    #     model = st.radio(
    #         "Choose a model",
    #         ("gpt-5-mini", "gpt-5-nano")
    #     )
    # # if use_other_provider:
    # #     model=st.radio("Choose Gemini/Cohere API",("gemini-2.0-flash-exp"))
    # if use_other_provider:
    #     provider_choice = st.radio("Choose Provider", ("Gemini", "Cohere"))
    #     if provider_choice == "Gemini":
    #         model = st.radio("Choose Gemini Model", ("gemini-2.0-flash-exp",))
    #     else:
    #         model = st.radio("Choose Cohere Model", ("command", "command-light", "command-nightly"))
cohere_api_key = st.secrets["Cohere_API_Key"]

openai_api_key  = st.secrets["API_KEY"]
Gemini_api_key=st.secrets["Gemini_API_Key"]
# def read_pdf(file):
#     pdf_reader = PyPDF2.PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() or ""
#     return text
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
        if use_other_provider:
            if model in ["gemini-2.0-flash-exp","gemini-2.5-pro"]:
                # Configure Gemini
                gemini_model = genai.GenerativeModel(model)
                
                # Extract content for Gemini
                prompt = messages[0]["content"]
                response = gemini_model.generate_content(prompt)
                # "models/gemini-2.5-pro"
                # def list_available_models():
                #     models = []
                #     for m in genai.list_models():
                #         if 'generateContent' in m.supported_generation_methods:
                #             models.append(m.name)
                #     return models
            
                # # Call this function to see available models
                # available_models = list_available_models()
                # st.write("Available Gemini models:", available_models)
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

            # elif model in ["command-r7b-12-2024","command-a-reasoning-08-2025","command", "command-light", "command-nightly"]:
            #     # Configure Cohere
            #     prompt = messages[0]["content"]
                
            #     def list_available_cohere_models():
            #         models = []
            #         try:
            #             model_list = cohere_client.models.list()
            #             # for model in model_list:
            #             #     models.append(model.name)
            #             return model_list
            #         except Exception as e:
            #             st.error(f"Error fetching Cohere models: {e}")
            #             return []
            #         # Display available models
            #     available_cohere_models = list_available_cohere_models()
            #     st.write("Available Cohere models:", available_cohere_models)

                
            #     try:
            #         response = cohere_client.chat(
            #         model=model,
            #         messages=[{"role": "user", "content": messages[0]["content"]}],  # Use messages array
            #         temperature=0.7,
            #         max_tokens=1000)
        
            #         st.write("Selected model:", model)
            #         # st.write("Response:", response.message.content[0].text)
            #         st.write("Response:", response.message[0])

                    
            #     except Exception as e:
            #         st.error(f"Error calling Cohere API: {str(e)}")
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
        # if use_other_provider and model == "gemini-2.0-flash-exp":
        #     # Configure Gemini
        #     genai.configure(api_key=Gemini_api_key)
        #     # gemini_model = genai.GenerativeModel('gemini-2.5-flash-002')
        #     gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

            
        #     # Extract content for Gemini
        #     prompt = messages[0]["content"]
        #     response = gemini_model.generate_content(prompt)
        #     def list_available_models():
        #         models = []
        #         for m in genai.list_models():
        #             if 'generateContent' in m.supported_generation_methods:
        #                 models.append(m.name)
        #         return models
        
        #     # Call this function to see available models
        #     available_models = list_available_models()
        #     # for model in available_models:
        #     #     print(model)
        #     st.write(available_models)
        #     st.write(model)
        #     st.write(response.text)
            
                        

        # else:
        #     stream = client.chat.completions.create(
        #                 model=model,
        #                 messages=messages
        #             )
        #     st.write(model)
        #     st.write(stream.choices[0].message.content)

