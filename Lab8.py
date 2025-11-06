import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Lab 8 - Multimodal Chatbot", layout="wide")
st.title("🧠 Lab 8 - Document Based Chatbot (with Images & Audio)")
st.write("""
### Objectives:
- Interact with a multimodal AI (GPT-4o or GPT-4o-mini)  
- Ask questions about **images** or **audio**  
- Observe how the model reasons across multiple input types
""")

# -------------------------------
# API Key (use your OpenAI key)
# -------------------------------
openai.api_key = st.secrets.get("API_KEY", "")

# -------------------------------
# Multimodal Text + Image Function
# -------------------------------
# def analyze_image_with_text(prompt, image_url):
#     """
#     Sends text and image to the multimodal model using correct input types.
#     """
#     try:
#         response = openai.responses.create(
#             model="gpt-4o-mini",  # or "gpt-4o"
#             input=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "input_text", "text": prompt},
#                         {"type": "input_image", "image_url": image_url}
#                     ]
#                 }
#             ]
#         )
#         return response.output[0].content[0].text
#     except Exception as e:
#         return f"⚠️ Error: {str(e)}"
def analyze_audio_with_text(prompt, audio_url):
    """
    Downloads an audio file, transcribes it with Whisper, 
    then uses GPT (via Responses API) to reason over the transcript.
    """
    try:
        # Step 1: Download the audio file
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(audio_url, headers=headers, allow_redirects=True)

        if response.status_code != 200:
            return f"⚠️ Could not fetch audio file (HTTP {response.status_code})"

        # Choose filename and save locally
        if audio_url.endswith(".wav"):
            filename = "temp_audio.wav"
        else:
            filename = "temp_audio.mp3"

        with open(filename, "wb") as f:
            f.write(response.content)

        # Step 2: Transcribe with Whisper
        with open(filename, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        # Step 3: Use the Responses API to analyze transcript
        gpt_response = openai.responses.create(
            model="gpt-4o-mini",   # or "gpt-4o"
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_text", "text": f"Audio Transcript:\n{transcript.text}"}
                    ]
                }
            ]
        )

        return gpt_response.output[0].content[0].text

    except Exception as e:
        return f"⚠️ Error processing audio: {str(e)}"


# -------------------------------
# Audio + Text Function (Optional)
# -------------------------------
def analyze_audio_with_text(prompt, audio_url):
    """
    Downloads an audio file, transcribes it with Whisper, then uses GPT for reasoning.
    """
    try:
        # Step 1: Download the audio file
        response = requests.get(audio_url, allow_redirects=True)
        if response.status_code != 200:
            return f"⚠️ Could not fetch audio file (HTTP {response.status_code})"

        # Detect the file type (use .mp3 by default)
        if audio_url.endswith(".wav"):
            filename = "temp_audio.wav"
        else:
            filename = "temp_audio.mp3"

        with open(filename, "wb") as f:
            f.write(response.content)

        # Step 2: Transcribe using Whisper
        with open(filename, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        # Step 3: Ask GPT about the transcript
        gpt_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nAudio Transcript:\n{transcript.text}"
                }
            ]
        )

        return gpt_response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error processing audio: {str(e)}"


# -------------------------------
# Streamlit Layout
# -------------------------------
st.header("🖼️ Image-Based Chat")

user_prompt = st.text_input("Enter your question about the image:")
image_url = st.text_input("Paste an image URL here:")

if image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Image from URL", use_container_width=True)
    except Exception:
        st.warning("Could not load the image. Please check the URL.")

if st.button("Analyze Image") and user_prompt and image_url:
    with st.spinner("Analyzing image with GPT..."):
        result = analyze_image_with_text(user_prompt, image_url)
    st.subheader("🤖 Model Response:")
    st.write(result)

# -------------------------------
# Optional: Audio Section
# -------------------------------
st.header("🎧 Audio-Based Chat (Optional)")

audio_prompt = st.text_input("Enter your question about the audio:")
audio_url = st.text_input("Paste an audio file URL (MP3):")

if st.button("Analyze Audio") and audio_prompt and audio_url:
    with st.spinner("Processing audio..."):
        audio_result = analyze_audio_with_text(audio_prompt, audio_url)
    st.subheader("🎙️ Model Response:")
    st.write(audio_result)

# -------------------------------
# Reflection Section
# -------------------------------
st.markdown("---")
# st.subheader("📝 Reflection & Discussion")
# st.write("""
# This lab demonstrates how multimodal models process different input types:
# - **Text + Image:** GPT-4o combines visual and linguistic reasoning to identify, describe, and infer context.
# - **Text + Audio:** Whisper transcribes the content; GPT then interprets or summarizes it.

# You can extend this app to handle multiple images, compare scenes, or even integrate PDFs and diagrams in future labs.
# """)
