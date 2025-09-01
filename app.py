import os
import streamlit as st
import pdfplumber
from pptx import Presentation
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# 初始化 OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("尚未設定 OPENAI_API_KEY。請在本機以環境變數或在部署平台的 Secrets 設定。")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化 ChromaDB
chroma_client = chromadb.PersistentClient(path="index")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="kmucer",
    embedding_function=openai_ef
)

# 輔助函數：讀取 PDF
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# 輔助函數：讀取 PPTX
def read_pptx(file):
    text = ""
    prs = Presentation(file)
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# Streamlit 介面
st.title("📚 KMUCer 助教 (ChromaDB 版)")

menu = st.sidebar.radio("選單", ["💬 助教對話", "🧱 建立知識庫"])

if menu == "🧱 建立知識庫":
    st.header("建立 / 更新教材知識庫")
    uploaded_files = st.file_uploader("上傳教材檔案 (PDF 或 PPTX)", type=["pdf", "pptx"], accept_multiple_files=True)
    if st.button("🚀 建立索引") and uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                text = read_pdf(file)
            else:
                text = read_pptx(file)
            # 切割文字成 chunk
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            for idx, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": file.name}],
                    ids=[f"{file.name}_{idx}"]
                )
        st.success("索引建立完成！")

elif menu == "💬 助教對話":
    st.header("向 KMUCer 提問")
    query = st.text_input("輸入你的問題：")
    if st.button("送出問題") and query:
        results = collection.query(query_texts=[query], n_results=3)
        context_chunks = [doc for doc in results["documents"][0]]
        context_text = "\n".join(context_chunks)

        system_prompt = """你是 KMUCer，高雄醫學大學醫藥暨應用化學系的課程助教。
        你是碩士班學姊，風格「專業 + 親切 + 搞笑」。
        請基於以下教材內容回答學生問題：
        {context}
        """.format(context=context_text)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        answer = completion.choices[0].message.content
        st.markdown("### 🗨️ KMUCer 回答")
        st.write(answer)
        st.markdown("### 📎 參考教材")
        st.write(results["metadatas"][0])
