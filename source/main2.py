import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_documents(path: str):
    if not os.path.exists(path):
        print(f"⚠️ Diretório não encontrado: {path}")
        return []

    pdfs = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]
    if not pdfs:
        print("⚠️ Nenhum PDF encontrado")
        return []

    print(f"📂 Processando {len(pdfs)} documentos...")
    start = time.time()

    def load(file_path):
        try:
            return PyPDFLoader(file_path).load()
        except Exception as e:
            print(f"⚠️ Erro em {os.path.basename(file_path)}: {e}")
            return []

    docs = []
    with ThreadPoolExecutor(max_workers=min(4, len(pdfs))) as executor:
        futures = [executor.submit(load, os.path.join(path, f)) for f in pdfs]
        for f in futures:
            docs.extend(f.result())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks criados em {time.time()-start:.1f}s")
    return chunks

def create_vector_store(documents):
    if not documents:
        return None

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dir_path = "./chroma_db"

    if os.path.exists(dir_path):
        print("♻️ Reutilizando vetores existentes")
        return Chroma(persist_directory=dir_path, embedding_function=embed)

    print("🧠 Criando vetores...")
    start = time.time()
    store = Chroma.from_documents(documents=documents, embedding=embed, persist_directory=dir_path)
    print(f"✅ Vetores criados em {time.time()-start:.1f}s")
    return store

def create_rag_agent():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "docs")

    docs = load_documents(path)
    if not docs:
        print("❌ Nenhum documento processado")
        return None

    store = create_vector_store(docs)
    if not store:
        return None

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    llm = Ollama(
        model="llama3.2:latest",
        temperature=0.2,
        num_ctx=4096,
        top_p=0.9,
        system="Você é um assistente especializado em documentos acadêmicos. Responda de forma completa e precisa, citando sempre as fontes. Se não souber, diga claramente que não encontrou a informação nos documentos."
    )

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

def main():
    print("⏳ Inicializando agente RAG...")

    try:
        agent = create_rag_agent()
        if not agent:
            raise RuntimeError("Falha na inicialização do agente")
    except Exception as e:
        print(f"❌ Erro crítico: {str(e)}")
        print("1. Verifique se há PDFs no diretório '../docs'")
        print("2. Instale dependências: pip install sentence-transformers pypdf")
        print("3. Para melhor desempenho: pip install -U langchain chromadb")
        return

    print("\n✅ Agente RAG pronto! (Digite 'sair' para terminar)")

    while True:
        try:
            query = input("\n>>> ").strip()
            if query.lower() in {"sair", "exit", "quit"}:
                break
            if not query:
                continue

            start = time.time()
            print("🔍 Buscando informações...", end="", flush=True)
            result = agent.invoke({"query": query})
            print(f"\r💡 Resposta ({time.time()-start:.1f}s):\n{result['result']}")

            if result['source_documents']:
                print("\n📚 Fontes encontradas:")
                src = {}
                for doc in result['source_documents']:
                    fname = os.path.basename(doc.metadata.get("source", "Documento"))
                    page = doc.metadata.get("page", "N/A")
                    if fname not in src:
                        src[fname] = set()
                    src[fname].add(page)
                for fname, pages in src.items():
                    pg = ", ".join(sorted(pages, key=lambda x: int(x) if x.isdigit() else 0))
                    print(f"  - {fname} (páginas: {pg})")
            else:
                print("\n⚠️ Nenhuma fonte relevante encontrada")

        except KeyboardInterrupt:
            print("\n🛑 Operação cancelada")
        except Exception as e:
            print(f"\n❌ Erro: {str(e)}")

    print("\n👋 Até logo!")

if __name__ == "__main__":
    main()
