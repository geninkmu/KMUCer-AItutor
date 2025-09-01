import os
import streamlit as st
import pdfplumber
from pptx import Presentation
from docx import Document
import pandas as pd
from striprtf.striprtf import rtf_to_text
from openai import OpenAI, RateLimitError
import numpy as np
import sqlite3
import json
import time

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

# 讀取 PDF
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# 讀取 PPTX
def read_pptx(file):
    text = ""
    prs = Presentation(file)
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# 讀取 DOCX
def read_docx(file):
    text = ""
    doc = Document(file)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# 讀取 TXT
def read_txt(file):
    return file.read().decode("utf-8")

# 讀取 CSV
def read_csv(file):
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="big5", errors="ignore")
    return df.to_string()

# 讀取 RTF
def read_rtf(file):
    raw_text = file.read().decode("utf-8", errors="ignore")
    return rtf_to_text(raw_text)

# 計算 embedding (含限速重試)
def get_embedding(text):
    for attempt in range(3):  # 最多重試 3 次
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except RateLimitError:
            st.warning(f"⚠️ API 被限速了，正在重試 ({attempt+1}/3)...")
            time.sleep(5)
    st.error("❌ 這段文字因為 API 限制無法建立 embedding，已跳過。")
    return None

# 儲存到 SQLite
def save_chunk(source, content, embedding):
    if embedding is not None:
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

# ---------------- App 開始 ----------------

st.title("📚 KMUCer 助教 (老師 / 學生模式)")

# 輸入角色密碼
role = st.sidebar.text_input("輸入角色密碼 (留空 = 學生模式)", type="password")

# 設定老師密碼
TEACHER_PASSWORD = "KMU2025"

# 模式判斷
if role == TEACHER_PASSWORD:
    mode = "teacher"
    st.sidebar.success("已進入：老師模式")
else:
    mode = "student"
    st.sidebar.info("目前是：學生模式")

# ---------------- 老師模式 ----------------
if mode == "teacher":
    menu = st.sidebar.radio("功能選單", ["💬 助教對話", "🧱 建立知識庫"])

    if menu == "🧱 建立知識庫":
        st.header("建立 / 更新教材知識庫")
        uploaded_files = st.file_uploader(
            "上傳教材檔案 (PDF, PPTX, DOCX, TXT, CSV, RTF)", 
            accept_multiple_files=True
        )
        if st.button("🚀 建立索引") and uploaded_files:
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = read_pdf(file)
                elif file.name.endswith(".pptx"):
                    text = read_pptx(file)
                elif file.name.endswith(".docx"):
                    text = read_docx(file)
                elif file.name.endswith(".txt"):
                    text = read_txt(file)
                elif file.name.endswith(".csv"):
                    text = read_csv(file)
                elif file.name.endswith(".rtf"):
                    text = read_rtf(file)
                else:
                    text = ""

                # 切割文字成 chunk (加大減少 API 請求數)
                chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
                for chunk in chunks:
                    emb = get_embedding(chunk)
                    save_chunk(file.name, chunk, emb)
            st.success("索引建立完成！")

# ---------------- 學生模式 & 對話功能 ----------------
st.header("向 KMUCer 提問")
query = st.text_input("輸入你的問題：")

if st.button("送出問題") and query:
    chunks = load_chunks()
    context_text = ""
    if chunks:
        query_emb = get_embedding(query)
        sims = [(cosine_similarity(query_emb, emb), source, content) for source, content, emb in chunks]
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[:3]
        context_text = "\n".join([s[2] for s in sims])
    else:
        st.info("📂 教材知識庫目前為空，將僅依靠大模型回答。")

    system_prompt = f"""你是 KMUCer，高雄醫學大學醫藥暨應用化學系的課程助教。
    你是碩士班學姊，風格「專業 + 親切 + 搞笑」。

    你有兩種知識來源：
    1. 教材知識庫（以下提供的內容） → 優先使用
    2. 你自己的專業知識（大模型本身） → 在教材沒有涵蓋時補充

    以下是教材內容（如果空白代表沒有相關教材）：
    {context_text}

    請結合這些知識，回答學生的問題。
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

    if context_text:
        st.markdown("### 📎 參考教材")
        st.write(context_text)
