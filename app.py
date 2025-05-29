# Invoice Extractor

import streamlit as st

# This must be the first Streamlit command
st.set_page_config(page_title="Invoice Extractor")

from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Use env var, don't hardcode in production

# Function to load Gemini Pro Vision model and get response
def get_gemini_response(prompt_text, image, context_prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt_text, image[0], context_prompt])
    return response.text

# Function to process uploaded image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# App Layout
st.title("Welcome to the Invoice Extractor")
st.header("Gemini Application")

user_prompt = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Tell me about the invoice")

# Context prompt for the model
input_prompt = """
You are an expert in understanding invoices. You will 
receive input images as invoices and you will have to
answer questions based on the input image.
"""

# Handle button click
if submit:
    if uploaded_file and user_prompt.strip():
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_prompt, image_data, user_prompt)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.warning("Please provide both a prompt and an image.")
