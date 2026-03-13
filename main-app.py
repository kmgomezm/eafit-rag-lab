import streamlit as st
import time
import base64
import numpy as np
import PyPDF2
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG vs LLM — EAFIT",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 RAG Playground — EAFIT Maestría Ciencia de Datos")
st.caption("Taller 03 · Comparación empírica: LLM Simple vs RAG Estándar vs RAG Optimizado")

# ─────────────────────────────────────────────
# Sidebar — Hiperparámetros
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Hiperparámetros")

    groq_api_key = st.text_input("🔑 Groq API Key", type="password")

    model_name = st.selectbox(
        "Model Select",
        ["llama3-70b-8192", "mixtral-8x7b-32768"],
        help="Llama-3-70b vs Mixtral-8x7b"
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    chunk_size = st.slider("Chunk Size (tokens aprox.)", 20, 2000, 500, 20)

    top_k = st.slider("Top-K (fragmentos recuperados)", 1, 10, 3)

    st.divider()
    st.markdown("**Embedding model:** `all-MiniLM-L6-v2`")
    st.markdown("**Vector Store:** FAISS (local)")

# ─────────────────────────────────────────────
# Carga de archivo
# ─────────────────────────────────────────────
st.subheader("📄 Cargar documento")
uploaded_file = st.file_uploader(
    "Sube un PDF o una imagen con texto (OCR vía Groq Vision)",
    type=["pdf", "png", "jpg", "jpeg", "webp"]
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_image(file_bytes: bytes, api_key: str, mime: str) -> str:
    """Usa llama-3.2-11b-vision-preview de Groq para OCR."""
    from groq import Groq
    client = Groq(api_key=api_key)
    b64 = base64.b64encode(file_bytes).decode()
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": "Extrae todo el texto visible en esta imagen, palabra por palabra."},
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


def build_vector_store(text: str, chunk_size: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(20, chunk_size // 10),
    )
    chunks = splitter.split_text(text)
    embeddings = get_embeddings()
    store = FAISS.from_texts(chunks, embeddings)
    return store, chunks


def cosine_sim_score(query: str, fragments: list[str]) -> float:
    embeddings = get_embeddings()
    q_vec = embeddings.embed_query(query)
    f_vecs = [embeddings.embed_query(f) for f in fragments]
    scores = cosine_similarity([q_vec], f_vecs)[0]
    return float(np.mean(scores))


def llm_simple(query: str, llm: ChatGroq) -> tuple[str, float]:
    start = time.time()
    response = llm.invoke([HumanMessage(content=query)])
    elapsed = time.time() - start
    return response.content, elapsed


def rag_query(
    query: str,
    store: FAISS,
    llm: ChatGroq,
    top_k: int,
    strict: bool = False,
) -> tuple[str, float, list[str], float]:
    start = time.time()
    docs = store.similarity_search(query, k=top_k)
    fragments = [d.page_content for d in docs]
    context = "\n\n---\n\n".join(fragments)

    system_content = (
        "Eres un asistente experto. Responde ÚNICAMENTE basándote en el contexto proporcionado. "
        "Si la información no está en el contexto, responde exactamente: 'No sé, esa información no está en el documento.'"
        if strict
        else "Eres un asistente experto. Usa el contexto proporcionado para responder con precisión."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Contexto:\n{context}\n\nPregunta: {query}"),
    ]
    response = llm.invoke(messages)
    elapsed = time.time() - start
    sim = cosine_sim_score(query, fragments)
    return response.content, elapsed, fragments, sim


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

document_text = None

if uploaded_file is not None:
    if not groq_api_key:
        st.warning("⚠️ Ingresa tu Groq API Key en el sidebar para continuar.")
    else:
        file_bytes = uploaded_file.read()
        mime_type = uploaded_file.type

        with st.spinner("Procesando documento..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                document_text = extract_text_from_pdf(file_bytes)
                st.success(f"✅ PDF procesado — {len(document_text)} caracteres extraídos.")
            else:
                document_text = extract_text_from_image(file_bytes, groq_api_key, mime_type)
                st.success(f"✅ Imagen procesada con OCR — {len(document_text)} caracteres extraídos.")

        with st.expander("📝 Ver texto extraído"):
            st.text(document_text[:3000] + ("..." if len(document_text) > 3000 else ""))

# ─────────────────────────────────────────────
# Zona de pregunta y comparación
# ─────────────────────────────────────────────
st.divider()
st.subheader("💬 Hacer una pregunta")

query = st.text_input("Escribe tu pregunta sobre el documento:")

run_btn = st.button("🚀 Comparar respuestas", disabled=(not query or not groq_api_key or document_text is None))

if run_btn:
    llm = ChatGroq(api_key=groq_api_key, model=model_name, temperature=temperature)

    # Construir vector store con chunk_size del sidebar (optimizado)
    with st.spinner("Generando embeddings y construyendo vector store..."):
        store_optimized, _ = build_vector_store(document_text, chunk_size)
        # RAG estándar usa chunk_size=500 por defecto
        store_default, _ = build_vector_store(document_text, 500)

    col1, col2, col3 = st.columns(3)

    # ── Columna 1: LLM Simple ──
    with col1:
        st.markdown("### 🤖 LLM Simple")
        st.caption("Inferencia sin contexto (Zero-shot)")
        with st.spinner("Generando..."):
            ans1, t1 = llm_simple(query, llm)
        st.write(ans1)
        st.metric("⏱ Tiempo de respuesta", f"{t1:.2f}s")
        st.metric("📐 Similitud coseno", "N/A (sin RAG)")

    # ── Columna 2: RAG Estándar ──
    with col2:
        st.markdown("### 📚 RAG Estándar")
        st.caption("RAG con parámetros por defecto (chunk=500, top-k=3)")
        with st.spinner("Recuperando y generando..."):
            ans2, t2, frags2, sim2 = rag_query(query, store_default, llm, top_k=3, strict=False)
        st.write(ans2)
        st.metric("⏱ Tiempo de respuesta", f"{t2:.2f}s")
        st.metric("📐 Similitud coseno (media)", f"{sim2:.4f}")
        with st.expander("🔍 Fragmentos recuperados"):
            for i, f in enumerate(frags2, 1):
                st.markdown(f"**Fragmento {i}:**\n> {f[:300]}...")

    # ── Columna 3: RAG Optimizado ──
    with col3:
        st.markdown("### ⚡ RAG Optimizado")
        st.caption(f"RAG con chunk={chunk_size}, top-k={top_k}, temp={temperature}")
        with st.spinner("Recuperando y generando..."):
            ans3, t3, frags3, sim3 = rag_query(query, store_optimized, llm, top_k=top_k, strict=True)
        st.write(ans3)
        st.metric("⏱ Tiempo de respuesta", f"{t3:.2f}s")
        st.metric("📐 Similitud coseno (media)", f"{sim3:.4f}")
        with st.expander("🔍 Fragmentos recuperados"):
            for i, f in enumerate(frags3, 1):
                st.markdown(f"**Fragmento {i}:**\n> {f[:300]}...")

# ─────────────────────────────────────────────
# Fase 4 — Análisis conceptual (st.expander)
# ─────────────────────────────────────────────
st.divider()
st.subheader("📊 Fase 4 — Análisis de Métricas y Conceptos")

with st.expander("1. Alucinación"):
    st.markdown("""
**¿En qué casos el LLM Simple inventa datos del documento?**

El LLM sin contexto puede alucinar fechas, nombres, cifras o afirmaciones
que sí existen en el documento pero que él desconoce. Al no tener acceso al
texto, genera respuestas plausibles basadas en su entrenamiento previo,
que pueden coincidir superficialmente pero diferir en detalles específicos
(e.g., inventar un autor o un porcentaje incorrecto). Con RAG, el modelo
ancla su respuesta a fragmentos reales, reduciendo drásticamente la alucinación.
""")

with st.expander("2. Inyección de Contexto"):
    st.markdown("""
**¿Cómo cambia la respuesta con un System Prompt estricto?**

En el **RAG Optimizado** se inyecta el system prompt:
> *"Si la información no está en el contexto, responde: 'No sé, esa información no está en el documento.'"*

Esto obliga al modelo a reconocer los límites de su conocimiento recuperado,
eliminando respuestas inventadas fuera del documento. Es especialmente útil
en entornos críticos (legal, médico, financiero) donde una respuesta incorrecta
es peor que ninguna.
""")

with st.expander("3. Fine-Tuning vs RAG"):
    st.markdown("""
**¿Por qué RAG es superior al Fine-Tuning para este ejercicio?**

| Criterio | Fine-Tuning | RAG |
|---|---|---|
| Costo | Alto (GPU, tiempo, datos) | Bajo |
| Actualización | Re-entrena todo el modelo | Solo actualiza el vector store |
| Transparencia | Opaco | Citable (muestra fragmentos) |
| Velocidad de despliegue | Días/semanas | Minutos |

Para documentos dinámicos o privados, RAG es la solución práctica y eficiente.
El fine-tuning tiene sentido cuando el dominio requiere cambiar el *estilo o comportamiento*
del modelo, no solo su conocimiento.
""")

with st.expander("4. Transformer vs No-Transformer (Embeddings)"):
    st.markdown("""
**¿Los embeddings dependen de arquitectura Transformer?**

Sí. El modelo `all-MiniLM-L6-v2` es un **Transformer encoder** basado en BERT.
Usa el mecanismo de **self-attention** para generar representaciones contextuales
densas de cada fragmento de texto. Sin Transformer, modelos como Word2Vec o TF-IDF
generan embeddings estáticos que no capturan el contexto semántico completo,
lo que reduce la calidad de la recuperación en RAG.
""")
