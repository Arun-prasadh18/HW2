import streamlit as st
from openai import OpenAI
import PyPDF2
# Show title and description.
st.title("📄My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = openai_api_key = st.secrets["API_KEY"]

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
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt","pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )
    document=None
    if uploaded_file and question:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
        # Process the uploaded file and question.
        # document = uploaded_file.read().decode()
        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API.
            # stream = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages,
            #     stream=True,
            # )
            
            # models = [
            # "gpt-3.5-turbo",
            # "gpt-4.1",
            # "gpt-5-chat-latest",
            # "gpt-5-nano"
            # ]

            # # Stream the response to the app using `st.write_stream`.
            # # st.write_stream(stream)
            # results = {}
            # import time
            # for model in models:
            #     start = time.time()
            #     response = client.chat.completions.create(
            #         model=model,
            #         messages=messages
            #     )
            #     end = time.time()
                
            #     answer = response.choices[0].message.content
            #     duration = round(end - start, 2)
                
            #     results[model] = {"answer": answer, "time": duration}

            # # Display results
            # for model, data in results.items():
            #     st.subheader(f"Model: {model}")
            #     st.write(f"⏱ Time: {data['time']}s")
            #     st.write(f"📝 Answer: {data['answer']}")
            

            models = [
            "gpt-3.5-turbo",
            "gpt-4.1",
            "gpt-5-nano",
            "gpt-5-chat-latest"
            ]

            INPUT_PRICES = {
                "gpt-3.5-turbo": 0.0005,
                "gpt-4.1": 0.002,
                "gpt-5-nano": 0.00005,
                "gpt-5-chat-latest": 0.00125
            }
            

            # CACHED_INPUT_PRICES = {
            #     "gpt-3.5-turbo": 0.0005,   # no cached pricing mentioned, same as input
            #     "gpt-4.1": 0.0005,
            #     "gpt-5-nano": 0.000005,
            #     "gpt-5-chat-latest": 0.000125
            # }

            OUTPUT_PRICES = {
                "gpt-3.5-turbo": 0.0015,
                "gpt-4.1": 0.008,
                "gpt-5-nano": 0.0004,
                "gpt-5-chat-latest": 0.01
            }

            results = {}
            import time
            for model in models:
                start = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                end = time.time()
                
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                # cached_tokens = getattr(usage, "cached_tokens", 0)  # handle cached tokens if API returns it
                
                cost = ((prompt_tokens) / 1000) * INPUT_PRICES[model] + \
                    (completion_tokens / 1000) * OUTPUT_PRICES[model]
                
                results[model] = {
                    "answer": response.choices[0].message.content,
                    "time": round(end - start, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": round(cost, 6)
                }

            # Show results
            for model, data in results.items():
                st.subheader(f"Model: {model}")
                st.write(f" Time: {data['time']}s")
                st.write(f" Tokens: prompt={data['prompt_tokens']},  completion={data['completion_tokens']}")
                st.write(f" Estimated cost: ${data['cost']}")
                st.write(f" Answer: {data['answer']}")

# document = "Paste your syllabus here"
# question = "Is this course hard?"

# messages = [
#     {"role": "user", "content": f"Here's the syllabus:\n\n{document}\n\n---\n\n{question}"}
# ]