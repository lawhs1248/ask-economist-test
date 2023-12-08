import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import streamlit as st 

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
import chromadb
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain


openai_token = os.environ.get("OPENAI_TOKEN", "")
openai_endpoint = "https://mti-nerve-openai-us-east-2.openai.azure.com/"

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_API_BASE"] = openai_endpoint
os.environ["OPENAI_API_KEY"] = openai_token    

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",chunk_size=1)

dir="./chroma_store/"
vectordb = Chroma(persist_directory=dir,embedding_function=embeddings)

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The answer which is from, the pdf documents name, e.g. `BA_1Q2023.pdf`",
        type="string",
    )
]

def create_agent_chain():
    llm = AzureChatOpenAI(temperature=0, 
        verbose=True, 
        deployment_name="gpt-4",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm, vectordb.as_retriever()
        )
    #chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query, source_documents):
    matching_docs = vectordb.similarity_search(query)
    matching_source = vectordb.metadata['source'](source_documents)
    chain = create_agent_chain()
    answer = chain.run(input_documents=matching_docs, question=query, source=matching_source)
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="Ask Economist", page_icon=":robot:")
st.header("Ask Economist")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input, form_input))

