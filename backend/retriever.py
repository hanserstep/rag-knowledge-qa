"""
混合检索引擎 - RAG 核心
======================
面试必考：BM25和向量检索的区别？为什么混合检索效果更好？

BM25（关键词匹配）：
- 基于词频统计，擅长精确匹配（如"Python报错"）
- 缺点：不理解语义，"电脑坏了"="计算机故障" 匹配不到

向量检索（语义匹配）：
- 将文本转为向量，计算余弦相似度
- 擅长语义相近的匹配（"电脑坏了"能找到"计算机故障"）
- 缺点：可能召回语义相关但不精确的结果

混合检索 = BM25 + 向量，各取 top_k，去重合并 → 互补优势
"""

import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from typing import List, Dict


class HybridRetriever:
    def __init__(self):
        # 向量模型：BGE-M3（支持中英文，8192维度→压缩到1024）
        # 国内源：从 ModelScope 加载（不再走 HF）
        self.embed_model = SentenceTransformer(
            os.path.expanduser("~/.cache/modelscope/BAAI/bge-m3"),
            device="cpu"  # M4 Mac 用 "mps" 更好
        )

        # Cross-Encoder 重排序模型
        self.rerank_model = CrossEncoder(
            os.path.expanduser("~/.cache/modelscope/BAAI/bge-reranker-v2-m3"),
            device="cpu"
        )

        # ChromaDB 持久化
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # BM25 索引（内存存储）
        self.bm25_index: Dict[str, BM25Okapi] = {}
        self.bm25_docs: Dict[str, List[str]] = {}

    # ---- 向量化 ----
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """将文本块转为向量，normalize后用于余弦相似度计算"""
        embeddings = self.embed_model.encode(
            chunks,
            normalize_embeddings=True,  # L2归一化，直接用点积算相似度
            show_progress_bar=False
        )
        return embeddings.tolist()

    # ---- 向量库操作 ----
    def add_to_vector_db(self, doc_id: str, chunks: List[str], embeddings: List[List[float]]):
        """写入 ChromaDB"""
        collection_name = f"doc_{doc_id}"
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass

        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 余弦距离
        )

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
        )

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量检索：在各文档集合中搜索"""
        query_embedding = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        all_results = []
        for coll in self.chroma_client.list_collections():
            results = coll.query(query_embeddings=[query_embedding], n_results=top_k)
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                all_results.append({"content": doc, "score": float(dist)})

        all_results.sort(key=lambda x: x["score"])
        return all_results[:top_k]

    # ---- BM25 操作 ----
    def add_to_bm25(self, doc_id: str, chunks: List[str]):
        """建立 BM25 索引（中文需要先分词）"""
        tokenized = [list(jieba.cut(chunk)) for chunk in chunks]
        self.bm25_index[doc_id] = BM25Okapi(tokenized)
        self.bm25_docs[doc_id] = chunks

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 关键词检索"""
        tokenized_query = list(jieba.cut(query))
        all_results = []
        for doc_id, bm25 in self.bm25_index.items():
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-top_k:][::-1]
            for idx in top_indices:
                if scores[idx] > 0:
                    all_results.append({
                        "content": self.bm25_docs[doc_id][idx],
                        "score": float(scores[idx])
                    })
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # ---- 混合检索 ----
    def hybrid_search(self, hyde_query: str, original_query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索策略（面试重点）：
        1. HyDE查询 → 向量检索（语义）
        2. 原始查询 → BM25检索（关键词）
        3. 合并去重 → 按来源标记
        4. 返回去重后的结果

        为什么用两个不同的query？
        - HyDE query 语义丰富，适合向量
        - 原始 query 保留关键词，适合 BM25
        """
        vector_results = self.vector_search(hyde_query, top_k=top_k)
        bm25_results = self.bm25_search(original_query, top_k=top_k)

        # 合并去重（按内容前30字符去重）
        seen = set()
        merged = []
        for r in vector_results + bm25_results:
            key = r["content"][:30]
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return merged[:top_k * 2]  # 返回多一点，给重排序用

    # ---- Cross-Encoder 重排序 ----
    def rerank(self, query: str, docs: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Cross-Encoder 重排序（面试重点）：
        - Bi-Encoder（向量模型）：query和doc分别编码，快但精度一般
        - Cross-Encoder：query+doc拼接后一起编码，慢但精度高
        - 策略：先用Bi-Encoder粗略召回，再用Cross-Encoder精排

        BGE-Reranker 输出 logit 值，越高越相关
        """
        if not docs:
            return docs

        pairs = [[query, d["content"]] for d in docs]
        scores = self.rerank_model.predict(pairs)

        for i, d in enumerate(docs):
            d["rerank_score"] = float(scores[i])

        docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return docs[:top_n]

    # ---- 管理操作 ----
    def list_collections(self) -> List[str]:
        return [c.name for c in self.chroma_client.list_collections()]

    def delete_collection(self, doc_id: str):
        try:
            self.chroma_client.delete_collection(f"doc_{doc_id}")
        except:
            pass
        self.bm25_index.pop(doc_id, None)
        self.bm25_docs.pop(doc_id, None)
