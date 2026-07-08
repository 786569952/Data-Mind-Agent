"""
agents/evaluator.py
===================
基于 RAGAS 0.4 的 RAG 检索与生成质量评估。

评估指标：
    - faithfulness      : 忠实度，答案是否忠于检索上下文 (无幻觉)
    - answer_relevancy  : 答案相关性，答案是否切题
    - context_precision : 上下文精度，检索结果中相关内容占比
    - context_recall    : 上下文召回，ground truth 信息是否被检索到

RAGAS 使用 LLM-as-Judge，evaluator LLM 可复用对话/Agent 的 LLM Provider：
    - bailian: ChatTongyi
    - ollama:  ChatOllama
"""
from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class EvalResult:
    """单次评估结果。"""
    query: str
    answer: str
    contexts: list[str]
    reference: str
    scores: dict  # metric_name -> score (0-1, -1 表示失败)

    def summary(self) -> str:
        lines = []
        for k, v in self.scores.items():
            if k.endswith("_error"):
                lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


def _wrap_llm(llm):
    """把 LangChain ChatModel 包装成 RAGAS 需要的 LangchainLLMWrapper。"""
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(llm)


async def _score_metric(metric, sample, wrapped_llm, wrapped_embeddings=None) -> float:
    """单个指标的异步评分。"""
    from ragas.run_config import RunConfig
    # RAGAS 0.4: 先设置 llm/embeddings，再 init (同步)，再 single_turn_ascore (async)
    metric.llm = wrapped_llm
    if wrapped_embeddings is not None and hasattr(metric, "embeddings"):
        metric.embeddings = wrapped_embeddings
    init_result = metric.init(RunConfig(timeout=120, max_retries=2))
    import inspect
    if inspect.isawaitable(init_result):
        await init_result
    score = await metric.single_turn_ascore(sample)
    return float(score)


def evaluate_rag(
    query: str,
    answer: str,
    contexts: list[str],
    reference: str = "",
    llm=None,
    embeddings=None,
    metrics: list[str] | None = None,
) -> EvalResult:
    """
    用 RAGAS 评估单条 RAG 样本 (同步接口，内部跑 async)。

    参数:
        query      : 用户问题
        answer     : Agent 生成的答案
        contexts   : 检索到的上下文片段列表
        reference  : 参考答案 (ground truth)，可选；为空则跳过 context_recall
        llm        : LangChain ChatModel，用作 evaluator
        embeddings : LangChain Embeddings，answer_relevancy 需要
        metrics    : 指定评估的指标列表，默认根据 reference 自动选择
    """
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    if llm is None:
        raise ValueError("必须传入 llm (LangChain ChatModel) 作为 evaluator")

    # 自动选择指标
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "context_precision"]
        if reference.strip():
            metrics.append("context_recall")

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    selected = [(name, metric_map[name]) for name in metrics if name in metric_map]

    # 构建 sample
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference or "(无参考答案)",
    )

    wrapped_llm = _wrap_llm(llm)

    # 包装 embeddings (answer_relevancy 需要)
    wrapped_embeddings = None
    if embeddings is not None:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # 异步逐指标评估
    async def _run_all():
        results = {}
        for name, metric in selected:
            try:
                score = await _score_metric(
                    metric, sample, wrapped_llm, wrapped_embeddings
                )
                results[name] = score
            except Exception as e:
                results[name] = -1.0
                results[f"{name}_error"] = str(e)[:300]
        return results

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已有事件循环 (Streamlit 环境) — 新建线程跑
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                scores = pool.submit(
                    lambda: asyncio.run(_run_all())
                ).result()
        else:
            scores = loop.run_until_complete(_run_all())
    except RuntimeError:
        scores = asyncio.run(_run_all())

    return EvalResult(
        query=query,
        answer=answer,
        contexts=contexts,
        reference=reference,
        scores=scores,
    )
