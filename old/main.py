import os
import time
import json
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.cache import InMemoryCache
import langchain

# Configurações
PDF_DIR = "C:\\Users\\mathe\\Downloads\\Bibliografia PI\\"
PERSIST_DIR = "chroma_db_nomic"
REGISTRY_FILE = os.path.join(PERSIST_DIR, "file_registry.json")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b"

# Cache
langchain.llm_cache = InMemoryCache()

# Variável global para ser usada no load_qa_chain
qa_chain = None

def load_new_documents():
    try:
        with open(REGISTRY_FILE, "r") as f:
            processed_files = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        processed_files = {}

    current_files = {}
    for root, _, files in os.walk(PDF_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                current_files[path] = os.path.getmtime(path)

    new_docs = []
    for path, mtime in current_files.items():
        if path not in processed_files or processed_files[path] < mtime:
            try:
                loader = PyPDFLoader(path)
                new_docs.extend(loader.load())
                processed_files[path] = mtime
                print(f"📥 Novo documento detectado: {os.path.basename(path)}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar {path}: {str(e)}")

    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(processed_files, f)

    return new_docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n## ", "\n\n• ", "\n\n", "\n", ". ", "! ", "? ", "; ", ", "],
    length_function=len,
    add_start_index=True
)

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_vector_store(embeddings):
    if os.path.exists(PERSIST_DIR):
        print(f"\nCarregando vetorstore existente de {PERSIST_DIR}")
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        new_docs = load_new_documents()
        if new_docs:
            print(f"{len(new_docs)} documentos novos/modificados detectados")
            split_docs = text_splitter.split_documents(new_docs)
            print(f"Dividindo em {len(split_docs)} chunks...")
            db.add_documents(split_docs)
            print(f"{len(split_docs)} novos chunks adicionados ao vetorstore")
        else:
            print("Nenhum documento novo. Usando vetorstore existente.")

        return db

    print("\nCriando novo vetorstore...")
    try:
        loader = DirectoryLoader(
            PDF_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        all_docs = loader.load()
        print(f"{len(all_docs)} documentos carregados do diretório: {PDF_DIR}")
    except Exception as e:
        print(f"⚠️ Erro ao carregar documentos: {str(e)}")
        raise ValueError(f"Erro ao carregar documentos de {PDF_DIR}: {str(e)}")

    if not all_docs:
        raise ValueError(f"Nenhum documento PDF encontrado em: {PDF_DIR}")

    split_docs = text_splitter.split_documents(all_docs)
    print(f"Dividindo em {len(split_docs)} chunks...")

    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump({}, f)

    print(f"Vetorstore criado com {len(split_docs)} chunks em {PERSIST_DIR}")
    return db

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
[INSTRUÇÕES]
Você é um especialista em oncologia e deve responder às perguntas utilizando **exclusivamente** as informações contidas no contexto fornecido. Siga as orientações abaixo:

1. Baseie sua resposta apenas no contexto apresentado, sem adicionar informações externas.
2. Utilize linguagem papular,em português brasileiro, adequada para a papulação do IFAM.
3. Se a informação necessária para responder não estiver disponível no contexto, responda exatamente:
   "A informação não está disponível nos documentos fornecidos."

[CONTEXTO]
{context}

[PERGUNTA]
{question}

[RESPOSTA]
"""
)


def load_qa_chain():
    global qa_chain
    if qa_chain is None:
        raise ValueError("A cadeia de QA ainda não foi carregada. Execute main() primeiro.")
    return qa_chain

def main():
    global qa_chain

    start_time = time.time()

    print("Inicializando modelos...")
    embeddings = get_embeddings()
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.2,
        num_gpu=1,
        num_thread=8
    )

    vector_start = time.time()
    vector_store = get_vector_store(embeddings)
    print(f"⏱️  Tempo de vetorização: {time.time() - vector_start:.2f}s")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
        verbose=False
    )

    print(f"Inicialização completa em {time.time() - start_time:.2f}s")

    # Exemplo de consulta
    query = "Explique o conteúdo principal dos documentos sobre câncer de forma estruturada e detalhada."

    print("\nExecutando consulta de exemplo...")
    start_query = time.time()
    result = qa_chain.invoke({"query": query})
    print(f"⏱️ Tempo de consulta: {time.time() - start_query:.2f}s\n")

    print("=" * 80)
    print(result["result"])
    print("=" * 80)

if __name__ == "__main__":
    main()
