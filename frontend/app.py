"""
RAG 知识库前端 - Streamlit
==========================
使用方法：streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG 知识库问答", layout="wide")
st.title("📚 RAG 智能知识库问答系统")

API = "http://localhost:8000"

# ---- 侧边栏：文档管理 ----
with st.sidebar:
    st.header("📄 文档管理")
    uploaded = st.file_uploader("上传文档", type=["pdf", "docx", "txt", "md"])
    if uploaded:
        with st.spinner("解析入库中..."):
            resp = requests.post(
                f"{API}/api/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue())}
            )
            if resp.ok:
                data = resp.json()
                st.success(f"✅ {data['message']}")
            else:
                st.error("上传失败")

    st.divider()
    if st.button("🔄 刷新文档列表"):
        st.rerun()

    try:
        docs = requests.get(f"{API}/api/documents").json()
        if docs["documents"]:
            st.write("**已入库文档：**")
            for d in docs["documents"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.code(d, language=None)
                with col2:
                    if st.button("删除", key=f"del_{d}"):
                        requests.delete(f"{API}/api/documents/{d}")
                        st.rerun()
        else:
            st.info("暂无文档，请上传")
    except:
        st.warning("后端未启动")

# ---- 主区域：问答 ----
st.subheader("💬 知识库问答")

col1, col2, col3 = st.columns(3)
with col1:
    use_hyde = st.checkbox("🧠 HyDE 策略", value=True,
                            help="用LLM生成假设答案辅助检索，提升召回率")
with col2:
    use_rerank = st.checkbox("🎯 Cross-Encoder 重排序", value=True,
                              help="对检索结果精排，提升Top-3准确率")
with col3:
    top_k = st.slider("检索数量", 1, 10, 5)

# 对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # 流式请求
            resp = requests.post(
                f"{API}/api/qa",
                json={
                    "question": prompt,
                    "top_k": top_k,
                    "use_hyde": use_hyde,
                    "use_rerank": use_rerank,
                    "stream": True
                },
                stream=True
            )

            sources_shown = False
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if data["type"] == "sources":
                        with st.expander("📎 参考来源"):
                            for i, s in enumerate(data["data"], 1):
                                st.caption(f"[{i}] {s}...")
                        sources_shown = True
                    elif data["type"] == "token":
                        full_response += data["data"]
                        placeholder.markdown(full_response + "▌")
                    elif data["type"] == "done":
                        placeholder.markdown(full_response)
                    elif data["type"] == "error":
                        st.error(data["data"])
        except Exception as e:
            st.error(f"连接后端失败：{e}")
            full_response = "⚠️ 无法连接到后端服务，请确认已启动 `python backend/main.py`"

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
