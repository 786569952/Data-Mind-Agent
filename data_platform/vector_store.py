"""
data_platform/vector_store.py
=============================
向量数据库操作 (ChromaDB 本地持久化)。

模拟数据平台 DW 层：清洗后的数据被向量化并写入向量库，供 Agent 检索。
"""
from __future__ import annotations

import os
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 持久化目录
DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vector_db",
)
DEFAULT_COLLECTION = "data_mind_kb"


def docs_from_records(records: list[dict[str, Any]]) -> list[Document]:
    """将 etl.build_documents 产生的记录列表转为 LangChain Document。"""
    return [
        Document(page_content=r["page_content"], metadata=r["metadata"])
        for r in records
    ]


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """从文档列表构建 (并持久化) 向量库。"""
    os.makedirs(persist_directory, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def load_vector_store(
    embeddings: Embeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """加载已持久化的向量库。"""
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def reset_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:
    """清空指定 collection 的持久化数据 (重建索引前调用)。"""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_directory)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    except Exception:
        # 兜底：删除整个目录
        if os.path.exists(persist_directory):
            import shutil

            shutil.rmtree(persist_directory)
