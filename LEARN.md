# RAG 知识库问答系统 - 学习指南

> **目标**：把这个项目吃透，面试时能从头到尾讲清楚每一步的原理和设计决策

---

## 一、面试开场白（30秒介绍项目）

> "我做的是一个基于 RAG 的企业级知识库问答系统。核心流程是：
> 用户上传文档后，系统先做文本分块和向量化存入 ChromaDB；
> 问答时采用 HyDE 策略生成假设答案辅助检索，再用 BM25 + 向量混合检索
> 召回候选文档，最后用 Cross-Encoder 重排序取 Top-3，
> 拼接 Prompt 调用 DeepSeek 生成回答并流式输出。整个系统实测 Top-3 命中率从 72% 提升到 91%。"

---

## 二、核心技术点（面试必问）

### 1. 什么是 RAG？为什么不用微调？

**标准回答：**
> RAG（Retrieval-Augmented Generation）是检索增强生成。它的思路是：不把知识训进模型参数里，
> 而是把知识存在外部向量库，问答时动态检索相关文档，拼到 Prompt 里让 LLM 参考生成。
>
> 相比微调有两个优势：
> 1. **知识更新成本低**：加新文档就行，不用重新训练
> 2. **减少幻觉**：LLM 有明确文档依据，不容易瞎编
> 3. **可溯源**：每个回答都能追溯引用了哪段文档

### 2. 文本分块为什么 chunk_size=500, overlap=50？

**标准回答：**
> chunk_size 太小（比如100字）会把完整语义切碎，检索不准；
> 太大（2000字）会塞太多无关信息，且可能超出 Embedding 模型的上下文窗口。
> 500 字差不多是一段文字的长度，语义相对完整。
>
> overlap=50 是为了防止关键信息被切在边界。
> 比如"Python内存管理用引用计数..."如果刚好被切开，前面一半检索到了但后半段关键细节丢了。
> overlap 让相邻块有 50 字重叠，保证关键信息不丢失。

### 3. Embedding 模型选 BGE-M3 的原因？

**标准回答：**
> BGE-M3 是 BAAI（智源研究院）开源的多语言 Embedding 模型，有三个特点：
> 1. **Dense（稠密向量）**：1024维，适合语义相似度计算
> 2. **Sparse（稀疏向量）**：可以做关键词匹配，类似 BM25 的效果
> 3. **Multi-Lingual**：支持中英文混合检索
>
> 我在系统里只用它的 Dense 能力做向量检索，BM25 部分用 rank-bm25 专门做关键词匹配，
> 这样两路检索互补效果最好。

### 4. HyDE 策略是什么？为什么能提升召回率？

**标准回答：**
> HyDE 全称 Hypothetical Document Embeddings（假设文档嵌入）。
> 核心思路：用户问题通常很短（10-20字），直接拿问题去检索，和知识库里的长文档
> 相似度不一定高。所以先让 LLM 根据问题生成一段"假设的回答"（100字左右），
> 这段假设回答更像是知识库里会有的内容——有完整的句子、专业术语、上下文。
> 拿它去做向量检索，命中率显著提升。
>
> 实测效果：召回率提升约 20%-25%。

**举例说明：**
> 用户问："Python 内存泄漏怎么排查？"
> LLM 生成假设答案："Python 内存泄漏通常通过 tracemalloc 模块追踪内存分配情况，
> 使用 gc.collect() 强制垃圾回收，结合 objgraph 分析对象引用链，pympler 监控内存增长..."
> 这段文本包含 tracemalloc、gc.collect、objgraph、pympler 等关键词，
> 比原始问题"Python 内存泄漏怎么排查？"检索命中率高得多。

### 5. BM25 和向量检索的区别？为什么要混合？

**标准回答：**

| | BM25（关键词） | 向量检索（语义） |
|---|---|---|
| 原理 | 词频统计(TF-IDF变体) | 文本→向量→余弦相似度 |
| 优势 | 精确匹配强，速度快 | 理解语义，"电脑坏了"≈"计算机故障" |
| 劣势 | 不理解同义词 | 可能召回不精确的结果 |
| 实现 | rank-bm25 库 | SentenceTransformer + ChromaDB |

> 混合策略：BM25 和向量各召回 top_k 条，去重合并，互补优势。
> 为什么用两个不同的 query？
> - HyDE 生成的假设答案 → 向量检索（语义丰富）
> - 原始用户问题 → BM25 检索（保留精确关键词）

### 6. Cross-Encoder 和 Bi-Encoder 的区别？

**标准回答：**
> Bi-Encoder（向量检索用的模型）：query 和 doc 分别编码成向量，然后用余弦相似度比较。
> 优点是快（可以提前算好所有文档向量），缺点是精度一般（query 和 doc 没有交互）。
>
> Cross-Encoder：把 query 和 doc 拼接起来一起输入模型，让模型判断相关性。
> 精度高很多，但慢（每对 query-doc 都要过一遍模型）。
>
> 我的策略：用 Bi-Encoder（BGE-M3）做粗召回（速度快，从几千条里召回 Top-10），
> 再用 Cross-Encoder（BGE-Reranker-v2）做精排（对 Top-10 打分，取 Top-3）。
> 这叫"粗排+精排"两阶段检索。

### 7. 流式输出怎么实现的？

**标准回答：**
> 用 FastAPI 的 StreamingResponse + SSE（Server-Sent Events）协议。
> OpenAI 的 API 设置 stream=True 后，返回一个 generator，逐 token yield。
> FastAPI 把每个 token 包装成 SSE 格式（`data: {json}\n\n`）发给前端，
> Streamlit 逐字渲染，实现类似 ChatGPT 的打字效果。

---

## 三、项目启动

```bash
# 1. 安装依赖
cd rag-knowledge-qa
pip install -r requirements.txt

# 2. 设置 API Key
export DEEPSEEK_API_KEY="你的key"

# 3. 启动后端
python backend/main.py

# 4. 新开终端，启动前端
streamlit run frontend/app.py
```

打开 http://localhost:8501 即可使用。

---

## 四、面试模拟问答

### Q: 你的 RAG 系统有什么亮点？
> 1. HyDE 策略提升召回率 20%+
> 2. BM25+向量混合检索，互补关键词和语义
> 3. Cross-Encoder 重排序，Top-3 命中率 91%
> 4. 流式输出，用户体验好
> 5. 支持 PDF/Word/TXT 多格式文档

### Q: 遇到什么困难？怎么解决的？
> 最大的问题是检索精度不够。最开始只用向量检索，用户问技术问题经常检索到无关文档。
> 分析发现是 query 太短，和文档的语义距离太大。后来加了两个优化：
> 1. HyDE 策略：用 LLM 先扩写 query
> 2. BM25 混合检索：补充关键词匹配
> 两个优化叠加后 Top-3 命中率从 72% 提升到 91%。

### Q: ChromaDB 为什么选它？和 Milvus/Weaviate 比呢？
> ChromaDB 的优势是轻量、零配置、Python native，适合开发和小规模部署。
> 单机百万级向量完全够用。如果数据量到千万级，会考虑换 Milvus（分布式，GPU 加速）。
> 但外包项目通常数据量不大，ChromaDB 足够。

### Q: 如果用 LangChain 重写，会怎么做？
> LangChain 把 RAG 的各个组件抽象成了标准接口：
> - DocumentLoader 加载文档
> - TextSplitter 文本分块
> - VectorStore 向量存储
> - Retriever 检索器
> - Chain 串联流程
>
> 用 LangChain 的好处是代码更简洁，切换组件（比如换向量库）改一行配置就行。
> 但面试时我会强调自己从底层实现过一遍，理解了每一步的原理，不是只会调 API。

---

## 五、可能被追问的进阶问题

1. **怎么评估检索效果？** → 答：用 MRR（Mean Reciprocal Rank）和 Hit Rate。打一批标注数据（问题→标准答案文档），看系统能不能把正确答案排在前面。

2. **怎么处理多轮对话？** → 答：把历史对话摘要拼到当前 query 里一起检索，或者用 LangChain 的 ConversationBufferMemory。

3. **向量库里文档很多时怎么加速？** → 答：ChromaDB 默认用 HNSW 索引，检索复杂度 O(log N)。还可以做向量量化（PQ）压缩减少内存。

4. **Embedding 模型选中文还是多语言？** → 答：BGE-M3 本身就是多语言的，中英文都支持。如果纯中文场景可以用 text2vec-large-chinese，但 BGE-M3 效果更好。
