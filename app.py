import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from main import cargar_documentos, crear_vectorstore, crear_cadena_qa
from langchain.chat_models import ChatOpenAI

# Cargar variables desde .env (OPENAI_API_KEY)
load_dotenv()

st.set_page_config(
    page_title="Claudia · Labmédico",
    page_icon="🤖",
    layout="wide",
)

# Logo institucional
st.sidebar.image("static/labmedico.jpg", use_container_width=True)

st.title("🤖 Chatea con Claudia (herramienta en construcción Labmédico)")
st.caption("Trabajando únicamente con los PDFs existentes en la carpeta `data/` (o la que elijas).")

# ── Sidebar: Configuración ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    # 1) lee primero de .env / entorno
    default_key = os.environ.get("OPENAI_API_KEY", "")
    # 2) si existen secrets (en Cloud), los usa
    try:
        default_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    api_key = st.text_input(
        "OPENAI_API_KEY",
        type="password",
        value=default_key,
        help="Se leerá de .env/local o de Secrets (en la nube).",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    data_dir = st.text_input(
        "📁 Carpeta de PDFs",
        value="data",
        help="Ruta relativa o absoluta. Debe contener archivos .pdf.",
    )
    rebuild = st.button("🔧 Reconstruir índice")

st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────
def folder_signature(folder: str) -> str:
    """Firma basada en nombres y mtimes de los PDFs para cachear/invalidar el índice."""
    p = Path(folder)
    if not p.exists():
        return "missing"
    parts = []
    for f in sorted(p.glob("*.pdf")):
        try:
            parts.append(f"{f.name}:{int(f.stat().st_mtime)}")
        except Exception:
            parts.append(f"{f.name}:0")
    return "|".join(parts) if parts else "empty"


@st.cache_resource(show_spinner=True)
def build_pipeline(signature: str, folder: str):
    """
    Construye vectorstore y cadena QA usando tus funciones de main.py.
    Usa la firma (signature) para invalidar la caché cuando cambian los PDFs.
    """
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")
    pdfs = list(p.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No hay PDFs en '{folder}'. Agrega archivos .pdf.")

    # Info rápida para debug
    st.write("📁 PDFs detectados en la carpeta:")
    for f in pdfs[:50]:  # muestra hasta 50 para no saturar
        st.write("•", f.name)
    if len(pdfs) > 50:
        st.write(f"… y {len(pdfs) - 50} más.")

    documentos = cargar_documentos(folder)
    vectorstore = crear_vectorstore(documentos)

    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

    cadena_qa = crear_cadena_qa(llm, vectorstore)
    return cadena_qa


# ── Preparar índice desde la carpeta elegida ───────────────────────────────────
sig = folder_signature(data_dir)
if rebuild:
    sig = sig + ":force"

try:
    cadena_qa = build_pipeline(sig, data_dir)
    num_pdfs = len(list(Path(data_dir).glob("*.pdf")))
    st.success(f"✅ Índice listo desde '{data_dir}'. PDFs detectados: {num_pdfs}")
except Exception as e:
    st.error(f"No se pudo preparar el índice: {e}")
    st.stop()

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hola, soy Claudia. Puedo ayudarte con los procedimientos, "
                "instructivos y documentos del laboratorio que están en la carpeta seleccionada. "
                "Puedes hacer preguntas generales como *«explícame la prueba de ciclaje»* "
                "o específicas como *«dame el paso a paso para la prueba de ciclaje»*."
            ),
        }
    ]

# Historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Escribe tu pregunta…")
if prompt:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Falta OPENAI_API_KEY. Configúrala en la barra lateral, .env o Secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            try:
                out = cadena_qa({"query": prompt})
                answer = out.get("result", "")
                sources = out.get("source_documents", []) or []
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.stop()

        st.markdown(answer or "Sin respuesta.")
        if sources:
            with st.expander("📚 Ver fuentes"):
                for i, doc in enumerate(sources, start=1):
                    meta = getattr(doc, "metadata", {}) or {}
                    src = meta.get("source") or meta.get("file_path") or "desconocido"
                    st.markdown(f"**Fuente {i}:** `{src}`")
                    try:
                        st.caption(
                            doc.page_content[:500]
                            + ("…" if len(doc.page_content) > 500 else "")
                        )
                    except Exception:
                        pass

    st.session_state.messages.append({"role": "assistant", "content": answer})
