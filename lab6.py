import requests
import streamlit as st
import openai
import json

st.title("AI Fact-Checker + Citation Builder"
         )
def fact_check_claim(user_claim):
    client = openai.OpenAI(api_key=st.secrets["API_KEY"])

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": """
    You are a factual verification assistant.
    For any given claim, search the web for credible sources and return a JSON object with:
        - claim
        - verdict: True / False / Partly True
        - explanation
        - sources
    """},
            {"role": "user", "content": user_claim}
        ],
        tools=[{"type": "web_search"}],
        text = {
    "format": {
        "type": "json_schema",
        "name": "fact_check_response",
        "schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "verdict": {
                    "type": "string",
                    "enum": ["True", "False", "Partly True"]
                },
                "explanation": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["claim", "verdict", "explanation", "sources"],
            "additionalProperties": False
        },
        "strict": True
    }
}
    )

    return response.output_text


user_claim = st.text_input("Enter a factual claim:")

if st.button("Check Fact"):
    with st.spinner("Verifying..."):
        result = fact_check_claim(user_claim)
        st.write(result)
