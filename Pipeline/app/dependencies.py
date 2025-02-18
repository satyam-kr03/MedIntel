from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Initialize vector store
vectorstore = Chroma(collection_name="rag-chroma", embedding_function=HuggingFaceEmbeddings())
store = InMemoryStore()
id_key = "doc_id"

# Initialize retriever
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

# Load LLM models
llm = Ollama(model="llama3.2", keep_alive=-1)
llava = Ollama(model="llava:7b-v1.6-mistral-q2_K", keep_alive=-1)
