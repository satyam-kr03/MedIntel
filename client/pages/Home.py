import streamlit as st
import streamlit.components.v1 as com
import aiohttp
import asyncio
import os
import uuid

st.set_page_config(page_title="MedIntel", page_icon="🩺", layout="wide")

BASE_URL = "https://cool-starfish-suitable.ngrok-free.app"

if "client_id" not in st.session_state:
    st.session_state.client_id = str(uuid.uuid4())

async def response_generator(prompt, client_id):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/generate/{client_id}?inp={prompt}") as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    yield chunk.decode("utf-8")
            else:
                yield "Error generating response"

async def upload_file(file, endpoint, client_id):
    form = aiohttp.FormData()
    form.add_field('file', file.getvalue(), 
                  filename=file.name,
                  content_type=file.type)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{BASE_URL}/{endpoint}/{client_id}', data=form) as response:
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
    st.markdown(colored_markdown(f"", "#007bff"),
                unsafe_allow_html=True)  # Blue color
    st.markdown(colored_markdown("How can I help you today ?", "#39A5A9").replace('</p>', ' style="font-size: 24px;"></p>'),
                unsafe_allow_html=True)  # Red color
except:

    st.markdown(colored_markdown(f"User", "#007bff"),
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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None

if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

# Display client ID in sidebar (optional, for debugging)
sidebar = st.sidebar
sidebar.markdown(f'<div style="text-align: center;"><img src="https://i.imgur.com/ngr2HSn.png" width="200"></div>', unsafe_allow_html=True)
sidebar.write("##")
sidebar.markdown("<h2 style='text-align: center;'>MedIntel v1.0</h2>", unsafe_allow_html=True)
sidebar.write("##")
sidebar.markdown(f"Session ID: {st.session_state.client_id}")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a pdf or image file",
        type=["pdf", "jpg", "png", "jpeg"],
        help="Upload a file to provide context for your queries."
    )

if uploaded_file is not None and (
    st.session_state.current_file is None or 
    uploaded_file.name != st.session_state.current_file
):
    with st.spinner("Processing file..."):
        endpoint = "upload_pdf" if uploaded_file.type == "application/pdf" else "upload_img"
        response = asyncio.run(
            upload_file(
                uploaded_file, 
                endpoint, 
                st.session_state.client_id
            )
        )
        
        st.session_state.upload_status = response
        st.session_state.file_uploaded = response == "File uploaded successfully"
        st.session_state.current_file = uploaded_file.name
        st.write(response)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat interface
if st.session_state.file_uploaded or uploaded_file is None:
    prompt = st.chat_input("What is up?", key="main_chat_input")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_placeholder = st.empty()
                response_container = []

                async def stream_response():
                    async for chunk in response_generator(
                        prompt, 
                        st.session_state.client_id
                    ):
                        response_container.append(chunk)
                        response_placeholder.markdown("".join(response_container))

                asyncio.run(stream_response())
                full_response = "".join(response_container)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Please wait until the file upload is complete before asking questions.")