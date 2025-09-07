import google.generativeai as genai
Gemini_api_key=st.secrets["Gemini_API_Key"]
# Configure with your API key
genai.configure(api_key=Gemini_api_key)

# List all available models
def list_available_models():
    models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            models.append(m.name)
    return models

# Call this function to see available models
available_models = list_available_models()
for model in available_models:
    print(model)
