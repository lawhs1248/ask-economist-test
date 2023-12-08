import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
import chromadb
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

openai_token = os.environ.get("OPENAI_TOKEN", "")
openai_endpoint = "https://mti-nerve-openai-us-east-2.openai.azure.com/"

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_API_BASE"] = openai_endpoint
os.environ["OPENAI_API_KEY"] = openai_token
    
embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002",chunk_size=1)

pdf_folder_path = "./input"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
client = chromadb.Client()
if client.list_collections():
    consent_collection = client.get_or_create_collection(name="ask-economist-collection",metadata={"hnsw:space": "cosine"})
else:
    print("Collection already exists")
vectordb = Chroma.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
    #collection_metadata={"hnsw:space": "cosine"}, 
    persist_directory="./chroma_store/"
)
vectordb.persist()