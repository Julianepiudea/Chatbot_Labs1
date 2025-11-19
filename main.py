import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

load_dotenv()


# ───────────────── Cargar y trocear documentos PDF ─────────────────────────────
def cargar_documentos(ruta_carpeta: str):
    """
    Carga todos los PDFs de la carpeta y los divide en chunks más pequeños
    para que cada llamada a embeddings sea manejable.
    """
    documentos = []

    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith(".pdf"):
            ruta_pdf = os.path.join(ruta_carpeta, archivo)
            loader = PyPDFLoader(ruta_pdf)
            documentos.extend(loader.load())

    if not documentos:
        raise RuntimeError(f"No se encontraron PDFs en la carpeta: {ruta_carpeta}")

    # Text splitter para trocear las páginas en fragmentos más pequeños
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   # un poco más largos para conservar contexto
        chunk_overlap=200, # solapamiento para no cortar ideas
    )
    chunks = splitter.split_documents(documentos)
    return chunks


# ───────────────────── Crear índice vectorial (FAISS) ──────────────────────────
def crear_vectorstore(documentos, batch_size: int = 64):
    """
    Crea un índice FAISS generando embeddings en lotes pequeños para
    evitar el error de 'max_tokens_per_request'.
    """
    if not documentos:
        raise ValueError("La lista de documentos está vacía; no se puede crear el índice.")

    # Modelo de embeddings de OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key,
    )

    # Primer lote para inicializar FAISS
    primeros = documentos[:batch_size]
    vectorstore = FAISS.from_documents(primeros, embeddi_
