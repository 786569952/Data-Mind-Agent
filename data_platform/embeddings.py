"""
data_platform/embeddings.py
===========================
向量化逻辑 (模拟数据平台 DW 层的索引构建)。

支持两种 Embedding provider:
    - bailian: 阿里百炼 DashScope (text-embedding-v1/v2/v3，需 API Key)
    - ollama:  本地 Ollama (bge-m3 / nomic-embed-text / mxbai-embed-large 等)
"""
from __future__ import annotations

from typing import Literal, Optional

from langchain_core.embeddings import Embeddings


def get_embeddings(
    api_key: str,
    model: str = "text-embedding-v3",
) -> "Embeddings":
    """
    构建 DashScope (阿里百炼) Embedding 客户端。

    参数:
        api_key: 阿里百炼 API Key
        model:   Embedding 模型名，可选 text-embedding-v1/v2/v3
    """
    from langchain_community.embeddings import DashScopeEmbeddings

    return DashScopeEmbeddings(
        model=model,
        dashscope_api_key=api_key,
    )


def get_ollama_embeddings(
    model: str = "bge-m3",
    base_url: str = "http://localhost:11434",
) -> "Embeddings":
    """
    构建 Ollama 本地 Embedding 客户端。

    常用模型 (需先 `ollama pull <model>`):
        - bge-m3              : 中文效果好，1024 维 (推荐)
        - nomic-embed-text    : 轻量，768 维
        - mxbai-embed-large   : 1024 维

    参数:
        model:    Ollama embedding 模型 tag
        base_url: Ollama 服务地址
    """
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )


def build_embeddings(
    provider: Literal["bailian", "ollama"],
    api_key: str = "",
    model: str = "",
    base_url: str = "http://localhost:11434",
) -> "Embeddings":
    """
    统一入口：按 provider 构建对应的 Embedding 客户端。

    - bailian: model 默认 text-embedding-v3，需要 api_key
    - ollama:  model 默认 bge-m3，需要本地 ollama serve
    """
    if provider == "ollama":
        return get_ollama_embeddings(
            model=model or "bge-m3",
            base_url=base_url,
        )
    return get_embeddings(
        api_key=api_key,
        model=model or "text-embedding-v3",
    )

