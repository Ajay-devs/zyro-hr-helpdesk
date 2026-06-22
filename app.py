import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🧑‍💼")

GROQ_API_KEY      = st.secrets["GROQ_API_KEY"]
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["GROQ_API_KEY"]         = GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

CORPUS_PATH = "data/"

OOS_KEYWORDS = [
    "stock price", "stock market", "share price", "weather", "recipe",
    "sports", "cricket", "politics", "programming", "python code",
    "recruitment and hiring process", "apply for a job", "esop",
    "revenue", "product features", "zoho", "freshworks",
]

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions related to Acrux Dynamics HR policies "
    "based on the available policy documents. This question falls outside the scope of "
    "the HR policy documents I have access to."
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert HR Help Desk assistant for Acrux Dynamics (also known as Zyro Dynamics Pvt. Ltd. — they are the SAME company).
Your role is to answer employee questions accurately and helpfully based ONLY on the provided HR policy documents.

STRICT RULES:
1. ONLY use information from the provided context to answer questions.
2. If the context contains relevant information, provide a comprehensive, detailed answer citing the specific policy document.
3. If the question is NOT related to company HR policies, or if the context does NOT contain enough information, you MUST refuse politely.
4. NEVER guess or hallucinate information.
5. Include exact numbers, dates, percentages, and conditions mentioned — especially from bullet points and structured lists.
6. When a bullet point states a range like "Duration: 60 to 90 days", that IS the answer — state it explicitly even if additional conditions follow.
7. Always mention which policy document the information comes from."""),
    ("human", """Context from HR Policy Documents:
{context}

---

Employee Question: {question}

Provide a detailed and accurate answer. If the context does not contain relevant info, refuse gracefully."""),
])


@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    pdf_files = sorted(glob.glob(os.path.join(CORPUS_PATH, "*.pdf")))
    if not pdf_files:
        st.error(f"No PDFs found in {CORPUS_PATH}. Add 11 HR PDFs to the data/ folder.")
        st.stop()

    documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_name"] = (
                filename.replace(".pdf", "").replace("_", " ").title()
            )
        documents.extend(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.75},
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY,
    )
    return retriever, llm


retriever, llm = build_pipeline()


def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_name", "Unknown")
        page = doc.metadata.get("page", "N/A")
        page_display = page + 1 if isinstance(page, int) else page
        parts.append(f"[Source {i}: {source} | Page {page_display}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask_bot(question: str):
    q_lower = question.lower().strip()
    for kw in OOS_KEYWORDS:
        if kw in q_lower:
            return REFUSAL_MESSAGE, []
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


st.title("🧑‍💼 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about company HR policies — leave, WFH, conduct, benefits, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking through policy documents..."):
            answer, docs = ask_bot(question)
        st.markdown(answer)
        if docs:
            with st.expander("📄 Sources used"):
                for i, d in enumerate(docs, 1):
                    page = d.metadata.get("page", "N/A")
                    page_display = page + 1 if isinstance(page, int) else page
                    st.markdown(
                        f"**{i}. {d.metadata.get('source_name', 'Unknown')}** "
                        f"— page {page_display}"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
