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
from langchain.chains import RetrievalQAWithSourcesChain

openai_token = os.environ.get("OPENAI_TOKEN", "")
openai_endpoint = "https://mti-nerve-openai-us-east-2.openai.azure.com/"

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_API_BASE"] = openai_endpoint
os.environ["OPENAI_API_KEY"] = openai_token    

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",chunk_size=1)
dir="./chroma_store/"
vectordb = Chroma(persist_directory=dir,
                  embedding_function=embeddings)


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What do you want to say to your PDF?"):
    # Display your message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add your message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # query ChromaDB based on your prompt. These results are ordered by similarity.
    q = vectordb.query(
        query_texts=[prompt]
    )
    results = q["documents"][0]

    prompts = []
    for r in results:
        # construct prompts based on the retrieved text chunks in results 
        prompt = "Please extract the following: " + prompt + "  solely based on the text below. Use an unbiased and journalistic tone. If you're unsure of the answer, say you cannot find the answer. \n\n" + r

        prompts.append(prompt)
    prompts.reverse()

    openai_res = AzureChatOpenAI.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "assistant", "content": prompt} for prompt in prompts],
        temperature=0,
    )

    response = openai_res["choices"][0]["message"]["content"]
    with st.chat_message("assistant"):
        st.markdown(response)

    # append the response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
