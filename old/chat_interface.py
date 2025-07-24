import streamlit as st
from main import get_embeddings, get_vector_store, custom_prompt, OllamaLLM
from langchain.chains import RetrievalQA
import os
import time

st.set_page_config(page_title="Chat com documentos", layout="wide")
st.title("🤖 Chat com seus documentos")

@st.cache_resource(show_spinner=False)
def create_qa_chain():
    # Inicializa embeddings e LLM
    embeddings = get_embeddings()
    llm = OllamaLLM(
        model="llama3:8b",
        temperature=0.2,
        num_gpu=1,
        num_thread=8
    )
    
    # Carrega vetorstore (index)
    vector_store = get_vector_store(embeddings)
    
    # Cria retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Cria cadeia QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False
    )
    
    return qa_chain

# Cria ou carrega o chain uma única vez
qa_chain = create_qa_chain()

# Histórico da conversa
if "history" not in st.session_state:
    st.session_state.history = []

# Input do usuário
user_input = st.chat_input("Digite sua pergunta sobre os documentos...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    
    with st.spinner("Pensando..."):
        response = qa_chain.invoke({"query": user_input})
        st.session_state.history.append({"role": "ai", "content": response["result"]})

# Exibe todo histórico na ordem
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
