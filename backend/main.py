"""
RAG 知识库后端 - FastAPI
-----------------------
面试重点：能讲清楚每个模块的职责和数据流

数据流：
  用户提问 → HyDE生成假设答案 → BM25+向量混合检索 → Cross-Encoder重排序
  → 拼接上下文 → LLM生成回答 → 流式返回
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import uuid

from retriever import HybridRetriever
from llm import DeepSeekLLM
from chunker import DocumentChunker

app = FastAPI(title="RAG Knowledge QA System", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- 全局组件（面试点：为什么用全局单例？答：向量模型加载慢，避免重复加载） ----
retriever = HybridRetriever()
llm = DeepSeekLLM()
chunker = DocumentChunker()


# ---- 1. 文档上传 & 入库 ----
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    面试点：文档处理流程
    1. 解析文档（PDF/Word/TXT）
    2. 文本分块（chunk_size=500, overlap=50）
    3. 生成向量并存入 ChromaDB
    4. 同时建立 BM25 索引
    """
    content = await file.read()
    suffix = file.filename.split(".")[-1].lower() if "." in file.filename else "txt"

    # 解析文档
    text = chunker.parse(content, suffix)
    if not text or len(text) < 50:
        raise HTTPException(400, "文档内容过短或无法解析")

    # 分块
    chunks = chunker.split(text, chunk_size=500, overlap=50)

    # 生成 embedding 并存入向量库
    doc_id = uuid.uuid4().hex[:12]
    embeddings = retriever.embed_chunks(chunks)
    retriever.add_to_vector_db(doc_id, chunks, embeddings)
    retriever.add_to_bm25(doc_id, chunks)

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "filename": file.filename,
        "message": f"成功入库 {len(chunks)} 个文本块"
    }


# ---- 2. 问答接口（核心） ----
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5          # 召回数量
    use_hyde: bool = True   # 是否启用 HyDE 策略
    use_rerank: bool = True # 是否启用重排序
    stream: bool = True     # 是否流式输出


@app.post("/api/qa")
async def ask(req: QuestionRequest):
    """
    面试关键流程（必须背熟每一步的原理）：
    Step1: HyDE → 让LLM生成假设答案，用假设答案去检索（提升召回25%）
    Step2: 混合检索 → BM25(关键词) + 向量(语义) 各取top_k，去重合并
    Step3: 重排序 → Cross-Encoder对每条文档打分，保留top_3
    Step4: 拼接 → System Prompt + 文档上下文 + 用户问题
    Step5: LLM生成 → 流式输出，带引用来源
    """
    # Step1: HyDE（Hypothetical Document Embeddings）
    search_query = req.question
    if req.use_hyde:
        search_query = llm.generate_hyde_query(req.question)

    # Step2: 混合检索
    docs = retriever.hybrid_search(search_query, req.question, top_k=req.top_k)

    # Step3: Cross-Encoder 重排序
    if req.use_rerank and len(docs) > 3:
        docs = retriever.rerank(req.question, docs, top_n=3)

    # Step4: 构建 Prompt
    context = "\n\n".join([f"[来源{i+1}] {d['content']}" for i, d in enumerate(docs)])
    prompt = llm.build_rag_prompt(req.question, context)

    # Step5: 流式生成
    if req.stream:
        return StreamingResponse(
            llm.stream_generate(prompt, docs),
            media_type="text/event-stream"
        )
    else:
        answer = llm.generate(prompt)
        return {"answer": answer, "sources": [d["content"][:100] for d in docs]}


# ---- 3. 文档列表 ----
@app.get("/api/documents")
async def list_documents():
    return {"documents": retriever.list_collections()}


# ---- 4. 删除文档 ----
@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    retriever.delete_collection(doc_id)
    return {"message": f"文档 {doc_id} 已删除"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
