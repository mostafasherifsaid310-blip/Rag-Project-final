import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from hf_api import query_hf_chat_model_stream

DB_PATH = "vectordb"

# -------------------- Embeddings --------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- Text Splitter (محسّن) --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,      # أدق من 500
    chunk_overlap=60     # تداخل منطقي
)

vectordb = None

# -------------------- Load Vectorstore --------------------
def load_vectorstore():
    global vectordb

    if os.path.exists(DB_PATH):
        try:
            vectordb = FAISS.load_local(
                DB_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            print("✅ Vectorstore Loaded")
        except Exception as e:
            print("❌ Error loading vectorstore:", e)
            vectordb = None

# -------------------- Build / Merge Vectorstore --------------------
def build_vectorstore(text):
    global vectordb

    chunks = splitter.split_text(text)

    if not chunks:
        print("⚠ No chunks created")
        return

    if vectordb is None:
        print("📦 Creating new vectorstore...")
        vectordb = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model
        )
    else:
        print("➕ Merging with existing vectorstore...")
        new_db = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model
        )
        vectordb.merge_from(new_db)

    vectordb.save_local(DB_PATH)
    print("✅ Vectorstore Updated")

# -------------------- Retrieval (MMR احترافي) --------------------
def retrieve_docs(query):

    if vectordb is None:
        return []

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 15
        }
    )

    return retriever.get_relevant_documents(query)

# -------------------- RAG Pipeline --------------------
def rag_with_citations(question):

    if vectordb is None:
        yield "⚠ لا يوجد ملفات مرفوعة"
        return

    docs = retrieve_docs(question)

    print("\n🔎 Retrieved Docs:\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.page_content[:200]}")
        print("------")

    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        yield "المعلومة غير موجودة فى الملفات"
        return

    # دمج السياق
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
- If the context does not contain info, explicitly say:
"I cannot find this information in the provided documents."
- Do NOT answer based on general knowledge.

Context:
{context}

Question:
{question}
"""

    full_answer = ""

    # Streaming من النموذج
    for partial in query_hf_chat_model_stream(prompt):
        full_answer = partial
        yield full_answer

    # إزالة المصادر المكررة
    unique_sources = []
    seen = set()

    for doc in docs:
        snippet = doc.page_content[:120]
        if snippet not in seen:
            seen.add(snippet)
            unique_sources.append(snippet)

    citations = "\n\n📚 Sources:\n"

    for i, src in enumerate(unique_sources, 1):
        citations += f"[{i}] {src}...\n"

    yield full_answer + citations

# -------------------- Initialize --------------------
load_vectorstore()