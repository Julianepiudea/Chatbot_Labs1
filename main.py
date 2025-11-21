import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ──────────────── Cargar y trocear documentos PDF ─────────────────
def cargar_documentos(ruta_carpeta: str):
    documentos = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith(".pdf"):
            ruta_pdf = os.path.join(ruta_carpeta, archivo)
            loader = PyPDFLoader(ruta_pdf)
            documentos.extend(loader.load())

    if not documentos:
        raise RuntimeError(f"No se encontraron PDFs en la carpeta: {ruta_carpeta}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documentos)

# ──────────────── Crear índice vectorial (FAISS) ───────────────────
def crear_vectorstore(documentos, batch_size: int = 64):
    if not documentos:
        raise ValueError("La lista de documentos está vacía; no se puede crear el índice.")

    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key,
    )

    primeros = documentos[:batch_size]
    vectorstore = FAISS.from_documents(primeros, embeddings)

    for i in range(batch_size, len(documentos), batch_size):
        lote = documentos[i:i + batch_size]
        vectorstore.add_documents(lote)

    return vectorstore

# ──────────────── Crear cadena de preguntas y respuestas ─────────────
def crear_cadena_qa(llm, vectorstore):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Eres Claudia, una asistente especializada en documentos técnicos de laboratorio.
Tu tarea es responder con precisión y claridad usando solo el contenido del documento.

Si la pregunta está relacionada con un procedimiento, resume los pasos completos en orden.
Si no encuentras información, responde: “No se encuentra en los documentos proporcionados.”

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain





