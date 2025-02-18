import streamlit as st
import streamlit.components.v1 as com
import aiohttp
import asyncio
import os

st.set_page_config(page_title="OPDx", page_icon="🩺", layout="wide")

BASE_URL = "https://cool-starfish-suitable.ngrok-free.app"

async def response_generator(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/generate?inp={prompt}") as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    yield chunk.decode("utf-8")  # Decode chunk to text
            else:
                yield "Error generating response"

async def upload_file(file, endpoint):
    form = aiohttp.FormData()
    form.add_field('file', file.getvalue(), 
                  filename=file.name,
                  content_type=file.type)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{BASE_URL}/{endpoint}', data=form) as response:
            if response.status == 200:
                return "File uploaded successfully"
            return "Error uploading file"
def colored_markdown(text: str, color: str):
    return f'<p class="title" style="background-color: #fff; color: {color}; padding: 5px; line-height: 1">{text}</p>'

sidebar = st.sidebar
col1, col2 = st.columns([1, 3])

# Display Lottie animation in the first column
with col1:
    com.iframe("https://lottie.host/embed/5899ceed-3498-4546-8ebf-b25561f40002/Xnif8r8nZ4.json", height=400, width=950)

try:
    st.markdown(colored_markdown(f"Luv", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9"),
                unsafe_allow_html=True)  # Red color
except:

    st.markdown(colored_markdown(f"Luv", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9"),
                unsafe_allow_html=True)  # Red color

st.markdown(
    """
    <style>
        .title {
            text-align: left; 
            font-size: 60px;
            margin-bottom: 0px;
            padding-bottom: 5px;
            display: inline-block;  
        }
        .subtitle {
            text-align: left;
            font-size: 24px;
            color: #333333; 
            margin-top: 5px; 
        }
        .square-container {
            display: flex;
            flex-wrap: wrap;
        }
        .square {
            width: 150px;
            height: 150px;
            background-color: #36A5A9;
            margin: 10px;
            margin-top: 30px;  
            margin-bottom: 50px;  
            color: #ffffff;
            padding: 10px;
            text-align: left;
            font-size: 14px;
            line-height: 1.5;
            border-radius: 16px;
            position: relative;  /* Enable relative positioning for image */
        }
        .square-image {
            position: absolute;  /* Make image absolute within square */
            bottom: 5px;  /* Position image at bottom */
            right: 5px;  /* Position image at right */
            width: 20px;
            height: 20px;
        }
        .input-container {
            display: flex;
            align-items: center;
            position: relative;
            margin-top: 20px;
        }
        .input-text {
            flex: 1;
            height: 40px;
            padding: 10px;
            font-size: 16px;
            border-radius: 12px;
            border: 1px solid #ccc;
            margin-right: 10px;
        }
        .button-container {
            display: flex;
            gap: 0px;
        }
        .button {
            
            height: 40px;
            width: 40px;
            margin: 0px;
            padding: 0px;
            display: flex;
            justify-content: center;
            align-items: center;
            # background-color: #39A5A9;
            color: #fff;
            border-radius: 8px;
            cursor: pointer;
        }
        .button svg {
            width: 24px;
            height: 24px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

uploaded_file = st.file_uploader(
    "Upload a pdf or image file",
    type=["pdf", "jpg", "png", "jpeg"],
    help="Upload a file to ask related questions."
)

if uploaded_file is not None and not st.session_state.file_uploaded:
    with st.spinner("Uploading file..."):
        if uploaded_file.type == "application/pdf":
            response = asyncio.run(upload_file(uploaded_file, "upload_pdf"))
        else:
            response = asyncio.run(upload_file(uploaded_file, "upload_img"))

        st.session_state.upload_status = response  # Store upload status
        st.session_state.file_uploaded = response == "File uploaded successfully"  # Mark as uploaded
        st.write(response)

# Prevent input until file upload is successful
if st.session_state.file_uploaded or uploaded_file is None:
    prompt = st.chat_input("What is up?", key="main_chat_input")  # Assign a unique key

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_placeholder = st.empty()
                response_container = []  # Use a list to store streamed content (mutable)

                async def stream_response():
                    async for chunk in response_generator(prompt):
                        response_container.append(chunk)  # Append each chunk
                        response_placeholder.markdown("".join(response_container))  # Update UI

                asyncio.run(stream_response())

                full_response = "".join(response_container)  # Convert list to full string

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Please wait until the file upload is complete before asking questions.")


sidebar = st.sidebar
sidebar.markdown(f'<img src="https://i.imgur.com/ngr2HSn.png" width="200">', unsafe_allow_html=True)
sidebar.write("##")
sidebar.markdown("<h2 style='text-align: center;'>User Dashboard</h2>", unsafe_allow_html=True)
sidebar.write("##")