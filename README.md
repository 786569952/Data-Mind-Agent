Data-Mind-Agent/
├── app.py                 # Streamlit 前端入口
├── agents/
│   ├── __init__.py
│   ├── rag_agent.py       # 核心 Agent 逻辑 (LangChain)
│   └── tools.py           # Agent 可用的工具 (如：执行 Python 代码、查询数据库)
├── data_platform/         # 模拟数据平台的核心模块
│   ├── __init__.py
│   ├── etl.py             # 数据清洗与预处理逻辑
│   ├── embeddings.py      # 向量化逻辑
│   └── vector_store.py    # 向量数据库操作
├── knowledge_base/        # 存放原始数据的文件夹
│   └── sample_data.csv
├── requirements.txt       # 依赖列表
├── README.md              # 项目介绍 (重点！)
└── .env                   # API Key 管理


Agent 框架: LangChain 或 AutoGen
*理由:* LangChain 是目前最标准的 RAG 框架，文档丰富，GitHub 上 Star 最多，适合展示；AutoGen 则侧重于多 Agent 对话，更有趣。
前端/UI: Streamlit (强烈推荐)
*理由:* 仅需 Python 代码即可生成漂亮的网页界面，无需学习 React/Vue，非常适合作为 MVP（最小可行性产品）上传 GitHub。
数据库/向量库: ChromaDB 或 FAISS (本地) / PostgreSQL + pgvector (如果模拟生产环境)
大模型: 通过 API 接入 DeepSeek (国产便宜强模型) 或 OpenAI。


项目架构设计（数据 Mass 平台视角）
为了让这个项目体现你对“数据平台”的理解，不要只做一个简单的问答机器人，而是将其设计为一个“数据资产管理 Agent”。

系统流程：

数据摄入: 模拟数据平台的 ODS 层，上传 CSV/Excel/文本数据。
数据处理: 进行清洗、切片，模拟数据平台的 ETL 过程。
向量化存储: 模拟数据平台的 DW（数据仓库）层，建立索引。
Agent 交互: 用户提问 -> Agent 生成检索计划 -> 获取数据 -> 生成答案。
