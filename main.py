import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
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
        chunk_size=1000,   # tamaño aproximado del trozo (en caracteres)
        chunk_overlap=200, # solapamiento entre trozos para no cortar ideas
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Primer lote para inicializar FAISS
    primeros = documentos[:batch_size]
    vectorstore = FAISS.from_documents(primeros, embeddings)

    # Resto de documentos por lotes
    for i in range(batch_size, len(documentos), batch_size):
        batch = documentos[i : i + batch_size]
        vectorstore.add_documents(batch)

    return vectorstore


# ───────────────────── Cadena de Preguntas y Respuestas ────────────────────────
def crear_cadena_qa(llm, vectorstore):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Eres Claudia, una asistente experta en documentos técnicos de laboratorio.
Responde de forma clara, concisa y profesional usando solo la información del contexto.
Si la respuesta no está en el contexto, di que no estás segura.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
""",
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


# ─────────────────────────── Modo consola (opcional) ───────────────────────────
def main():
    print("📚 Cargando documentos desde 'data'...")
    documentos = cargar_documentos("data")
    print(f"📄 Fragmentos de texto cargados: {len(documentos)}")

    print("🧠 Creando índice vectorial (FAISS)...")
    vectorstore = crear_vectorstore(documentos)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    cadena_qa = crear_cadena_qa(llm, vectorstore)

    print("🤖 Chatbot listo. Escribe tu pregunta (o 'salir'):")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "salir":
            break
        respuesta = cadena_qa({"query": pregunta})
        print("🤖:", respuesta["result"])
