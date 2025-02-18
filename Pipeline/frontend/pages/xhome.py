import streamlit as st
import requests
from io import BytesIO

# Define backend API URLs
UPLOAD_IMG_API = "https://cool-starfish-suitable.ngrok-free.app/upload_img"
GENERATE_API = "https://cool-starfish-suitable.ngrok-free.app/generate"
UPLOAD_PDF_API = "https://cool-starfish-suitable.ngrok-free.app/upload_pdf"

# Streamlit UI
st.title("Multimodal RAG Chatbot")

# File uploader for image and PDF
uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# Text input for user query
user_query = st.text_input("Ask something:")

if uploaded_img is not None:
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
    if st.button("Upload Image"):
        # Upload image to backend
        files = {"file": uploaded_img.getvalue()}
        response = requests.post(UPLOAD_IMG_API, files=files)
        if response.status_code == 200:
            st.write("Image uploaded successfully!")
        else:
            st.write("Error uploading image")

if uploaded_pdf is not None:
    st.write("PDF uploaded: ", uploaded_pdf.name)
    if st.button("Upload PDF"):
        # Upload PDF to backend
        files = {"file": uploaded_pdf.getvalue()}
        response = requests.post(UPLOAD_PDF_API, files=files)
        if response.status_code == 200:
            st.write("PDF uploaded successfully!")
        else:
            st.write("Error uploading PDF")

# Handle user query for generate response
if user_query:
    response = requests.post(GENERATE_API, json={"inp": user_query})
    if response.status_code == 200:
        st.write(f"Bot Response: {response.json().get('response')}")
    else:
        st.write("Error generating response")
