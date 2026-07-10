"""
agents/tools.py
===============
Agent 可用的工具集。

工具列表:
    - get_data_schema:  查看数据集结构 (列名 / 类型 / 行数)
    - get_data_stats:   查看数值列统计摘要
    - query_knowledge_base: 基于向量库的语义检索
    - filter_records:   按条件筛选记录
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma


def _get_df(ctx: dict[str, Any]) -> pd.DataFrame:
    df = ctx.get("df")
    if df is None:
        raise ValueError("当前没有可用的数据集，请先在「数据摄入」页上传并清洗数据。")
    return df


def _get_store(ctx: dict[str, Any]) -> Optional[Chroma]:
    return ctx.get("vector_store")


# 工具工厂：通过闭包绑定运行时上下文 (df / vector_store)
# 这样 Agent 调用工具时即可访问会话内的数据。


def make_tools(ctx: dict[str, Any]) -> list:
    """根据会话上下文构建工具列表。"""

    @tool
    def get_data_schema() -> str:
        """查看当前数据集的结构：列名、数据类型、总行数。无需参数。"""
        df = _get_df(ctx)
        lines = [f"总行数: {len(df)}", "列结构:"]
        for col in df.columns:
            lines.append(f"  - {col} ({df[col].dtype})")
        return "\n".join(lines)

    @tool
    def get_data_stats(columns: str = "") -> str:
        """
        查看数值列的统计摘要 (count/mean/std/min/max 等)。
        参数:
            columns: 可选，逗号分隔的列名；为空则返回所有数值列统计。
        """
        df = _get_df(ctx)
        # 容错: ReAct 可能传入 {"columns": "price"} 这样的 JSON 字符串
        cols_str = columns
        if cols_str and cols_str.strip().startswith("{"):
            import json as _json
            try:
                parsed = _json.loads(cols_str)
                if isinstance(parsed, dict):
                    cols_str = str(parsed.get("columns", "")) or ""
            except Exception:
                pass
        if cols_str.strip():
            cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            # 只保留真实存在的列
            cols = [c for c in cols if c in df.columns]
            if not cols:
                sub = df.select_dtypes(include="number")
            else:
                sub = df[cols].select_dtypes(include="number")
        else:
            sub = df.select_dtypes(include="number")
        if sub.empty:
            return "数据集中没有数值列。"
        return sub.describe().round(2).to_string()

    @tool
    def query_knowledge_base(query: str, top_k: int = 5) -> str:
        """
        基于向量库对清洗后的数据做语义检索。
        参数:
            query:   自然语言查询
            top_k:   返回的最相似记录数 (默认 5)
        """
        store = _get_store(ctx)
        if store is None:
            return "知识库尚未建立，请先在「数据摄入」页完成向量化。"
        docs = store.similarity_search(query, k=top_k)
        if not docs:
            return "未检索到相关记录。"
        # 保存检索结果到 ctx，供 RAGAS 评估使用
        ctx["_last_retrieved_contexts"] = [d.page_content for d in docs]
        ctx["_last_retrieval_query"] = query
        blocks = []
        for i, d in enumerate(docs, 1):
            blocks.append(f"[{i}] {d.page_content}")
        return "\n".join(blocks)

    @tool
    def filter_records(condition: str) -> str:
        """
        按 pandas query 语法筛选记录，返回前 20 条。
        参数:
            condition: pandas DataFrame.query 表达式，例如 "price > 1000 and region == '华东'"
        """
        df = _get_df(ctx)
        try:
            sub = df.query(condition)
        except Exception as e:
            return f"筛选表达式错误: {e}"
        if sub.empty:
            return "没有满足条件的记录。"
        return f"找到 {len(sub)} 条匹配记录，前 20 条:\n{sub.head(20).to_string()}"

    return [get_data_schema, get_data_stats, query_knowledge_base, filter_records]
