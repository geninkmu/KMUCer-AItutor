import os
import streamlit as st
import pdfplumber
from pptx import Presentation
from openai import OpenAI
import numpy as np
import sqlite3
import json

# 初始化 OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("尚未設定 OPENAI_API_KEY。請在本機以環境變數或在部署平台的 Secrets 設定。")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化 SQLite
DB_PATH = "kmucer.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content TEXT,
                embedding TEXT
            )''')
conn.commit()

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

# 計算 embedding
def get_embedding(text):
    while True:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except RateLimitError:
            st.warning("⚠️ API 請求太快，被限速了，等一下自動重試...")
            time.sleep(5)

# 儲存到 SQLite
def save_chunk(source, content, embedding):
    c.execute("INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)", 
              (source, content, json.dumps(embedding.tolist())))
    conn.commit()

# 從 SQLite 取所有 chunk
def load_chunks():
    c.execute("SELECT source, content, embedding FROM chunks")
    rows = c.fetchall()
    chunks = []
    for row in rows:
        source, content, emb_json = row
        emb = np.array(json.loads(emb_json))
        chunks.append((source, content, emb))
    return chunks

# 相似度計算 (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Streamlit 介面
st.title("📚 KMUCer 助教 (SQLite 儲存版)")

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
            for chunk in chunks:
                emb = get_embedding(chunk)
                save_chunk(file.name, chunk, emb)
        st.success("索引建立完成！")

elif menu == "💬 助教對話":
    st.header("向 KMUCer 提問")
    query = st.text_input("輸入你的問題：")
    if st.button("送出問題") and query:
        chunks = load_chunks()
        if not chunks:
            st.warning("⚠️ 尚未建立知識庫，請先上傳教材！")
        else:
            query_emb = get_embedding(query)
            # 找出最相似的三個 chunk
            sims = [(cosine_similarity(query_emb, emb), source, content) for source, content, emb in chunks]
            sims = sorted(sims, key=lambda x: x[0], reverse=True)[:3]
            context_text = "\n".join([s[2] for s in sims])

            system_prompt = f"""你是 KMUCer，高雄醫學大學醫藥暨應用化學系的課程助教。
            你是碩士班學姊，風格「專業 + 親切 + 搞笑」。
            請基於以下教材內容回答學生問題：
            {context_text}
            """

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
            for s in sims:
                st.write(f"- 來源: {s[1]} | 內容: {s[2][:100]}...")
