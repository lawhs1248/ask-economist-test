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

# Define the 'generate_response' function to send the user's message to the AI model 
# and append the response to the 'generated' list.
def generate_response(prompt, conversation_chain):
    try:
        result = conversation_chain(prompt)
        return result["answer"], ' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))
    except Exception as e:
        print(e)
        return "I am unable to get the response based on this question, please fine-tune it before retrying", ""

# The 'chat_click' function is defined to send the user's message to the AI model 
# and append the response to the conversation history.
def chat_click(user_chat_input, conversation_chain):
    if user_chat_input != '':
        answer, sources=generate_response(user_chat_input, conversation_chain)
        st.session_state['sources'] = []
        st.session_state['past'] = []
        st.session_state['answers'] = []
        st.session_state['sources'].append(sources)
        st.session_state['past'].append(user_chat_input)
        st.session_state['answers'].append(answer)

# Streamlit to set the page header and icon.
st.set_page_config(page_title="Ask Economist", page_icon=":robot:")
st.title("Ask Economist")
st.subheader('Notes:')
term_of_use = """- This application is a BETA version
- Data Security: Please only input information classified up to Official (Closed) / Non-Sensitive 
- Accuracy: Due to ongoing development and the nature of the AI language model, the results may generate inaccurate or misleading information
- Accountability: All output must be fact-checked, proof-read, and adapted as appropriate by officers for their work
- Feedback: If you have any suggestion to improve this application, please email: :blue[mti-do_helpdesk@mti.gov.sg]
"""
st.markdown(term_of_use)
# container for text box
container = st.container()
# container for chat history
response_container = st.container()

sample_qns = """Sample questions:
1. How can we boost Singapore's productivity?
2. Was the JGI scheme successful and if so, how was it successful? 
"""

with container:
    with st.form(key='my_form', clear_on_submit=True):
        st.subheader("Question: ")
        st.markdown(sample_qns)
        user_input = st.text_area("Question:", key='input', height=100, label_visibility="hidden")
        submit_button = st.form_submit_button(label='Send')

    
    if submit_button and user_input:
        chat_click(user_input, chain)

# The 'message' function is defined to display the messages in the conversation history.
if 'answers' in st.session_state:
    if st.session_state['answers']:
        with response_container:
            cols = st.columns(2)
            for i in range(len(st.session_state['answers'])):
                with cols[0]:
                    st.subheader("Question: ")
                    st.write(st.session_state['past'][i])
                    st.subheader("Answer: ")
                    st.write(st.session_state['answers'][i])
                with cols[1]:
                    st.subheader("Sources: ")
                    for index, source in enumerate(st.session_state['sources'][i].split(".pdf")):
                        if source:  # Check if source is not an empty string
                            source_parts = source.split("\\")  # Split by backslash
                            source_name = source_parts[-1]  # Take the last part after backslash
                            github_url = "https://github.com/lawhs1248/ask-economist-test/blob/main/input/"
                            st.write(index+1, source_name)
                            st.write(github_url + source_name.replace(' ', '%20') + '.pdf')
                                
        # send_survey_result(st.session_state.session_id, st.session_state.nerve_logger, st.session_state['credentials_correct'], user_input)