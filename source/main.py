import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# 1. Carrega e divide os PDFs
def load_documents(path: str):
    pdf_files = glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True)
    all_docs = []

    for file in pdf_files:
        try:
            loader = PyPDFLoader(file)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Erro ao carregar {file}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(all_docs)

# 2. Cria o banco vetorial
def create_vector_store(documents):
    embeddings = OllamaEmbeddings(model="llama3.2")
    return Chroma.from_documents(documents=documents, embedding=embeddings)

# 3. Cria o agente RAG
def create_rag_agent():
    documents = load_documents("..\\docs")
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = Ollama(model="llama3.2", temperature=0.1)

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# 4. Interface
def main():
    agent = create_rag_agent()
    print("🤖 Agente RAG com LLaMA 3.2 inicializado. Digite sua pergunta (ou 'sair' para terminar):")

    while True:
        query = input("\n>>> ")
        if query.lower() in ["sair", "exit"]:
            break

        result = agent({"query": query})
        print(f"\n🔍 Resposta:\n{result['result']}")
        print(f"\n📚 Fontes consultadas:")
        for i, doc in enumerate(result['source_documents'], 1):
            meta = doc.metadata
            print(f"{i}. {meta.get('source', 'desconhecida')} (página {meta.get('page', 'N/A')})")

if __name__ == "__main__":
    main()
