# 🔍 RAG 知识库问答系统

> 基于检索增强生成（RAG）的智能知识库问答系统，支持文档上传、混合检索、HyDE 策略与流式对话。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 核心特性

- **混合检索**：BM25（关键词匹配）+ 向量检索（语义理解），互补召回
- **HyDE 策略**：先让 LLM 生成假设答案，再用假设答案检索，提升相关性
- **Cross-Encoder 重排序**：BGE-Reranker-v2-m3 对召回结果精排
- **多格式文档解析**：支持 PDF、DOCX、TXT、Markdown
- **智能分块**：基于语义段落 + 滑动窗口，保留上下文连贯性
- **流式输出**：支持 SSE（Server-Sent Events）实时打字效果
- **来源追溯**：回答附带 `[来源N]` 引用标记

## 🏗️ 技术架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  FastAPI     │────▶│  DeepSeek   │
│  前端界面    │     │  后端服务     │     │  LLM API    │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼────┐ ┌─────▼──────┐
       │  ChromaDB   │ │ BM25  │ │CrossEncoder│
       │  向量数据库   │ │ 索引  │ │   重排序    │
       └─────────────┘ └───────┘ └────────────┘
              │
       ┌──────▼──────┐
       │  BGE-M3     │
       │  Embedding  │
       └─────────────┘
```

## 📦 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 后端框架 | FastAPI + Uvicorn | 高性能异步 API |
| 前端 | Streamlit | 快速构建交互式 UI |
| 向量数据库 | ChromaDB | 轻量级本地向量存储 |
| Embedding | BGE-M3 (1.9GB) | 多语言，1024 维 |
| 关键词检索 | BM25 (rank-bm25) | 经典信息检索算法 |
| 重排序 | BGE-Reranker-v2-m3 | Cross-Encoder 精排 |
| LLM | DeepSeek V3 | 高性能国产大模型 |
| 分词 | jieba | 中文分词（BM25） |
| 模型源 | ModelScope | 国内镜像加速下载 |

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 8GB+ 可用内存（加载 BGE-M3 模型需 ~4GB）

### 安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv && source venv/bin/activate

# 2. 安装依赖（国内用户用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置 API Key
echo 'DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx' > .env

# 4. 下载模型（国内用户从 ModelScope 加载自动完成）
# 首次运行会自动从 ModelScope 下载 BGE-M3 和 BGE-Reranker
```

### 启动

```bash
# 启动后端 API（默认 http://localhost:8000）
cd backend && python main.py

# 启动前端界面（默认 http://localhost:8501）
streamlit run app.py
```

## 🧪 测试

```bash
python test_rag.py
```

```
Chunker 模块:          6 pass, 0 fail ✅
BM25 检索:             2 pass, 0 fail ✅
LLM 模块 (HyDE/流式):  4 pass, 0 fail ✅
Retriever 模块:         2 pass, 0 fail ✅
FastAPI 路由:           3 pass, 0 fail ✅
─────────────────────────────────────
总计:                  17/17 ✅
```

## 📂 项目结构

```
rag-knowledge-qa/
├── backend/
│   ├── main.py          # FastAPI 主应用（路由 + 生命周期）
│   ├── chunker.py       # 文档解析与智能分块
│   ├── retriever.py     # 混合检索器（BM25 + Vector + Rerank）
│   └── llm.py           # LLM 客户端（HyDE + 流式/非流式生成）
├── app.py               # Streamlit 前端界面
├── test_rag.py          # 完整测试套件（17 用例）
├── LEARN.md             # 学习指南
└── requirements.txt     # Python 依赖
```

## 🔑 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文档（PDF/DOCX/TXT/MD） |
| POST | `/api/qa` | 问答接口（支持 HyDE + 重排序） |
| GET | `/api/documents` | 列出已上传的文档 |
| DELETE | `/api/documents/{doc_id}` | 删除指定文档 |

## 📊 实测性能

- **问答延迟**：~3.7s（含 HyDE 生成 + 混合检索 + 重排序 + LLM 回答）
- **文档解析**：PDF ≤ 2s, DOCX ≤ 1s
- **检索精度**：BM25 + Vector 混合 + CrossEncoder 三重保障

## 📝 License

MIT License
