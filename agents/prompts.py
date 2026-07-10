"""
agents/prompts.py
=================
Prompt 模板管理 + LLM 自动优化 (meta-prompting)。

提供三类 Prompt 的默认模板与变量:
    1. agent_system   : Agent 系统提示词 (bailian)
    2. react_template : ReAct 提示词 (ollama)
    3. retrieval_qa   : 检索测试页的问答 Prompt

支持:
    - render(template, **vars)  : 渲染变量预览
    - optimize_prompt(llm, ...) : LLM 自动分析并改写 Prompt
"""
from __future__ import annotations

from typing import Any

# ============ 默认 Prompt 模板 ============

DEFAULT_AGENT_SYSTEM = """你是一名「数据资产管理 Agent」，服务于一个模拟的数据平台。

你的能力:
1. 了解当前数据集的结构与统计特征 (通过 get_data_schema / get_data_stats)。
2. 按条件筛选记录 (通过 filter_records)。
3. 基于向量库做语义检索 (通过 query_knowledge_base)。

工作原则:
- 先用工具获取事实，再回答；不要凭空编造数据。
- 回答用中文，结构清晰，必要时给出数字依据。
- 涉及统计/汇总时，先调用 get_data_stats 获取真实统计量。
- 检索语义相关记录时使用 query_knowledge_base。
- 如果数据集或知识库不可用，请明确提示用户先去「数据摄入」页完成清洗与向量化。

当前会话上下文:
- 数据集: {has_data}
- 知识库: {has_kb}
- 模型: {provider} / {model}
"""

DEFAULT_REACT_TEMPLATE = """{system}

你可以使用以下工具:
{tools}

请严格按照以下格式回答:

Question: 用户的问题
Thought: 你应该怎么思考
Action: 要使用的工具名 (必须是 [{tool_names}] 之一)
Action Input: 工具的输入参数 (JSON 字符串)
Observation: 工具返回的结果
... (Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我已经获得足够信息
Final Answer: 给用户的最终答案

注意:
- Action 必须是工具名之一，不要加引号。
- Action Input 必须是合法 JSON。
- 开始吧！

Question: {input}
{agent_scratchpad}"""

DEFAULT_RETRIEVAL_QA = """请根据以下检索到的上下文回答问题。

上下文：
{context}

问题：{question}

回答 (简洁准确)："""

# 各模板支持的变量说明 (用于前端展示)
PROMPT_META: dict[str, dict[str, Any]] = {
    "agent_system": {
        "label": "Agent 系统提示词 (百炼)",
        "description": "百炼 Provider 使用的 system prompt，定义 Agent 角色与工作原则",
        "variables": {
            "has_data": "数据集状态 (已就绪/未加载)",
            "has_kb": "知识库状态 (已就绪/未建立)",
            "provider": "模型 Provider (bailian/ollama)",
            "model": "模型名称",
        },
    },
    "react_template": {
        "label": "ReAct 提示词 (Ollama)",
        "description": "Ollama Provider 使用的 ReAct 格式提示词",
        "variables": {
            "system": "系统提示词 (来自 agent_system)",
            "tools": "工具描述列表",
            "tool_names": "工具名列表 (逗号分隔)",
            "input": "用户输入",
            "agent_scratchpad": "Agent 中间步骤",
        },
    },
    "retrieval_qa": {
        "label": "检索问答 Prompt (检索测试页)",
        "description": "检索测试页基于上下文生成答案时使用的 Prompt",
        "variables": {
            "context": "检索到的上下文文本",
            "question": "用户问题",
        },
    },
}

# meta-prompt: 让 LLM 优化另一个 prompt
META_OPTIMIZE_PROMPT = """你是一名 Prompt 工程专家。请优化以下 Prompt 模板。

【当前 Prompt 模板】
---
{current_prompt}
---

【Prompt 用途】
{purpose}

【优化目标】
1. 指令更清晰、无歧义
2. 约束更明确，减少模型跑偏
3. 保留所有原有变量占位符 (用 {{variable}} 表示)，不要新增或删除变量
4. 中文表达，简洁有力

请直接输出优化后的完整 Prompt 模板，不要加任何解释或前后缀。"""


def render(template: str, **kwargs: Any) -> str:
    """安全渲染 Prompt 模板，未提供的变量保留原样。"""
    try:
        return template.format(**kwargs)
    except KeyError:
        # 逐个替换，缺失的保留 {name}
        result = template
        for k, v in kwargs.items():
            result = result.replace("{" + k + "}", str(v))
        return result


def optimize_prompt(
    llm,
    current_prompt: str,
    purpose: str,
) -> str:
    """
    用 LLM 自动优化 Prompt 模板 (meta-prompting)。

    参数:
        llm            : LangChain ChatModel
        current_prompt : 待优化的 Prompt 模板
        purpose        : Prompt 的用途说明
    返回:
        优化后的 Prompt 模板字符串
    """
    meta_prompt = META_OPTIMIZE_PROMPT.format(
        current_prompt=current_prompt,
        purpose=purpose,
    )
    result = llm.invoke(meta_prompt)
    optimized = result.content.strip() if hasattr(result, "content") else str(result).strip()
    # 去掉可能的 ``` 代码块包裹
    if optimized.startswith("```"):
        lines = optimized.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        optimized = "\n".join(lines)
    return optimized
