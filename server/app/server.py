import os
import uuid
import subprocess
import base64
import time
import glob
import shutil
import pytesseract
import threading

from pyngrok import ngrok
from typing import Dict, Any
from collections import OrderedDict
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

from unstructured.partition.auto import partition

from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore

@dataclass
class ClientStorage:
    vectorstore: Chroma
    docstore: InMemoryStore
    retriever: MultiVectorRetriever

class ClientStorageManager:
    def __init__(self, max_clients: int = 50):
        self.client_storage: OrderedDict[str, ClientStorage] = OrderedDict()
        self.embeddings = HuggingFaceEmbeddings()
        self.max_clients = max_clients
    
    def create_client_storage(self, client_id: str) -> ClientStorage:
        """Create new storage instances for a client, evicting least recently used if necessary."""
        if len(self.client_storage) >= self.max_clients:
            # Remove the least recently used client (first item in OrderedDict)
            lru_client_id, _ = self.client_storage.popitem(last=False)
            print(f"Evicted LRU client: {lru_client_id}")

        vectorstore = Chroma(
            collection_name=f"rag-chroma-{client_id}",
            embedding_function=self.embeddings,
        )
        docstore = InMemoryStore()
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",
        )
        
        storage = ClientStorage(vectorstore, docstore, retriever)
        self.client_storage[client_id] = storage
        self.client_storage.move_to_end(client_id, last=True)  # Mark as recently used
        return storage
    
    def get_client_storage(self, client_id: str) -> ClientStorage:
        """Get or create storage for a client, updating usage order."""
        if client_id not in self.client_storage:
            return self.create_client_storage(client_id)
        
        # Move accessed client to the end (most recently used)
        self.client_storage.move_to_end(client_id, last=True)
        return self.client_storage[client_id]

class ChatSession:
    def __init__(self, max_history: int = 30):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only the last max_history messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_formatted_history(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.history
        ])

# Initialize storage manager
storage_manager = ClientStorageManager()
chat_sessions: Dict[str, ChatSession] = {}

#import nltk
#nltk.download('averaged_perceptron_tagger')

# Set environment variables
os.environ["PATH"] += r";C:\Users\go39res\AppData\Local\miniconda3\envs\stuff\Library\bin"

# Configure Tesseract
TESSERACT_PATH = r"C:\Users\go39res\AppData\Local\Programs\Tesseract-OCR"
os.environ["PATH"] += os.pathsep + TESSERACT_PATH
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, "tesseract.exe")

print(pytesseract.get_tesseract_version())
print(shutil.which("tesseract"))

# Configure ngrok
ngrok.set_auth_token("2rr1W3HKR8YWgtA1UBoBuGI7vBM_4yPLBV8EibUReXoc6oV1d")

tunnel_config = {
    "addr": "8000",
    "hostname": "cool-starfish-suitable.ngrok-free.app",
    "proto": "http"
}

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000"
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

id_key = "doc_id"

# Initialize the models
llm = Ollama(
    model="llama3.2",
    stop=["###", "{", "Below is an instruction describing a task, paired with an input providing further context. Write a response that appropriately completes the request."],
    system="You are a knowledgeable AI assistant. For medical-related inquiries, provide responses aligned with clinical evidence, best practices, and ethical guidelines. For general queries, respond as a helpful AI assistant, ensuring clarity, accuracy, and relevance.",
    keep_alive=-1
)

llava = Ollama(model="llava:7b-v1.6-mistral-q2_K", keep_alive = -1)

# Test the models
print("Models loaded successfully")
print("Testing Llava: ")
print(llava("Hi"))
print("Testing Llama: ")
print(llm("Hi"))

# Utility functions
def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/upload_img/{client_id}")
async def up_img(client_id: str, file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        storage = storage_manager.get_client_storage(client_id)
        cleaned_img_summary = []
        
        # Read the uploaded image file
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Process image with LLaVA
        RES = llava(prompt="Provide a concise, factual summary of the image, capturing all the key visual elements and details you observe. Avoid speculative or imaginative descriptions not directly supported by the contents of the image. Focus on objectively describing what is present in the image without introducing any external information or ideas. Your summary should be grounded solely in the visual information provided.", images=[image_base64])
        cleaned_img_summary.append(RES)
        
        img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(cleaned_img_summary)
        ]
        
        storage.vectorstore.add_documents(summary_img)
        storage.docstore.mset(
            list(zip(img_ids, cleaned_img_summary))
        )
        
        return {"message": "200"}
    
    except Exception as e:
        print(e)
        return {"message": "404"}


@app.post("/generate/{client_id}")
async def generate_output(client_id: str, inp: str):
    storage = storage_manager.get_client_storage(client_id)

    if client_id not in chat_sessions:
        chat_sessions[client_id] = ChatSession()
    
    session = chat_sessions[client_id]
    
    #s = "You Are my personal doctor. You have to Remember my symptoms antecendants past history and try to ask me follow up questions whenever I tell you that I am not well"
    s = "You are an AI assistant"

    # Prompt template
    template = """Below is an instruction describing a task, paired with an input providing further context. Write a response that appropriately completes the request.

    {{ if .}}### Instruction: Respond based on the given context and chat history. If the inquiry is medical, provide an accurate and professional response as a healthcare professional. Otherwise, respond as a general AI assistant.{{ .System }}{{ end }} 

    Previous conversation:
    {chat_history}

    Current context: {context}
    Current question: {question}

    ### Response:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Add user message to history
    session.add_message("user", inp)

    # Modified RAG pipeline with chat history
    chain = (
        {
            "context": storage.retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: session.get_formatted_history()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    async def generate():
        response_chunks = []
        async for chunk in chain.astream(inp):
            response_chunks.append(chunk)
            yield chunk
        
        full_response = "".join(response_chunks)
        session.add_message("assistant", full_response)

    return StreamingResponse(generate(), media_type='text/event-stream')

@app.post("/upload_pdf/{client_id}")
async def post_pdf(client_id: str, file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        storage = storage_manager.get_client_storage(client_id)
        print("PDF processing started...")
        # Save the uploaded file
        filename = "uploaded_file.pdf"
        path = "./"
        
        content = await file.read()
        with open(path + filename, "wb") as f:
            f.write(content)

        # Rest of your original code remains exactly the same
        raw_pdf_elements = partition(
            filename = path + filename,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=path,
        )
        category_counts = {}
        
        print("PDF partitioned successfully...")

        for element in raw_pdf_elements:
            category = str(type(element))
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

        # Unique_categories will have unique elements
        # TableChunk if Table > max chars set above
        unique_categories = set(category_counts.keys())
        category_counts
        class Element(BaseModel):
            type: str
            text: Any

        # Categorize by type
        categorized_elements = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element)))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element)))

        print("PDF elements categorized successfully...")
        # Tables
        table_elements = [e for e in categorized_elements if e.type == "table"]
        # print(len(table_elements))

        # Text
        text_elements = [e for e in categorized_elements if e.type == "text"]
        # print(len(text_elements))

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text. \
        Give a concise summary of the table or text. Table or text chunk: {element} """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

        texts = [i.text for i in text_elements if i.text != ""]
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        print(text_summaries)
        # Apply to tables
        tables = [i.text for i in table_elements]
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

        # Add texts
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
        storage.vectorstore.add_documents(summary_texts)
        storage.docstore.mset(list(zip(doc_ids, texts)))

        # Add tables
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=s, metadata={id_key: table_ids[i]})
            for i, s in enumerate(table_summaries)
        ]
        storage.vectorstore.add_documents(summary_tables)
        storage.docstore.mset(list(zip(table_ids, tables)))

        IMG_DIR = './'

        # Use glob to match file paths
        image_files = glob.glob(f"{IMG_DIR}*.jpg")
        # print(image_files)
        cleaned_img_summary = []

        # Iterate over matched file paths
        for img in image_files:
            # Perform your operation here
            RES = llava(prompt="Provide a concise, factual summary of the image, capturing all the key visual elements and details you observe. Avoid speculative or imaginative descriptions not directly supported by the contents of the image. Focus on objectively describing what is present in the image without introducing any external information or ideas. Your summary should be grounded solely in the visual information provided." ,images=[str(image_to_base64(img))])
            cleaned_img_summary.append(RES)
            
        img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
        summary_img = [
            Document(page_content=s, metadata={id_key: img_ids[i]})
            for i, s in enumerate(cleaned_img_summary)
        ]
        print(cleaned_img_summary)
        
        storage.vectorstore.add_documents(summary_texts)
        storage.docstore.mset(list(zip(doc_ids, texts)))
        return {"message":"200"}
    except Exception as e:
        print(e)
        return {"message":"404"}

# main function to run the server
if __name__ == "__main__":
    import uvicorn
    try:
        tunnel = ngrok.connect(**tunnel_config)
        print(f"Ngrok tunnel established at: {tunnel.public_url}")
    except Exception as e:
        print(f"Error establishing ngrok tunnel: {e}")
        exit(1)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        ngrok.disconnect(tunnel.public_url)