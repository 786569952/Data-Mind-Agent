"""
agents/rag_agent.py
===================
核心 Agent 逻辑 (LangChain Agent + 多 provider)。

支持的 provider:
    - bailian: 阿里百炼 DashScope (通义千问 qwen-*，支持 native tool calling)
    - ollama:  本地 Ollama llama (llama3.1 / qwen2.5 / ...)

Agent 策略:
    - bailian:  create_tool_calling_agent (native tool_calls)
    - ollama:   create_react_agent (文本解析 ReAct，兼容性最好，因为
                langchain_ollama 0.3.x 的 bind_tools 不会把 tools 发送给
                Ollama /api/chat，native tool calling 不稳定)

流式输出:
    `arun_agent_stream` 基于 `astream_events(version="v2")` 真正逐 token
    产出事件，前端可据此渲染打字机效果 + 工具调用链。
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from agents.tools import make_tools


SYSTEM_PROMPT = """你是一名「数据资产管理 Agent」，服务于一个模拟的数据平台。

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

# ReAct 提示词 (用于 ollama provider)
REACT_TEMPLATE = """{system}

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


def build_llm(
    provider: Literal["bailian", "ollama"],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
):
    """
    构建对话大模型客户端。

    provider=bailian:
        model:    qwen-turbo / qwen-plus / qwen-max / qwen-long
        api_key:  DashScope API Key (必填)
    provider=ollama:
        model:    llama3.1 / llama3.2 / qwen2.5:7b ... (需本地 ollama pull)
        base_url: 默认 http://localhost:11434
    """
    if provider == "bailian":
        from langchain_community.chat_models import ChatTongyi

        return ChatTongyi(
            dashscope_api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            streaming=True,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
            top_p=top_p,
        )
    else:
        raise ValueError(f"未知 provider: {provider}")


def build_agent(
    ctx: dict[str, Any],
    provider: Literal["bailian", "ollama"],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> AgentExecutor:
    """
    构建数据资产管理 Agent。

    - bailian: create_tool_calling_agent (native tool_calls)
    - ollama:  create_react_agent (文本解析，兼容性好)
    """
    llm = build_llm(provider, model, api_key, base_url, temperature, top_p)
    tools = make_tools(ctx)
    tool_names = ", ".join(t.name for t in tools)

    has_data = "已就绪" if ctx.get("df") is not None else "未加载"
    has_kb = "已就绪" if ctx.get("vector_store") is not None else "未建立"
    system_text = SYSTEM_PROMPT.format(
        has_data=has_data, has_kb=has_kb, provider=provider, model=model,
    )

    if provider == "ollama":
        # ReAct: 用文本格式，避免依赖 native tool calling
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(
            system=system_text,
            tool_names=tool_names,
        )
        # tools 描述需要手动拼到 {tools} 占位符
        from langchain.tools.render import render_text_description
        prompt = prompt.partial(tools=render_text_description(tools))
        agent = create_react_agent(llm, tools, prompt)
    else:
        # bailian: native tool calling
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


def run_agent(
    executor: AgentExecutor,
    user_input: str,
    chat_history: list | None = None,
) -> dict[str, Any]:
    """非流式运行，返回 {answer, steps}。"""
    result = executor.invoke({
        "input": user_input,
        "chat_history": chat_history or [],
    })
    return {"answer": result.get("output", ""), "steps": _extract_steps(result)}


async def arun_agent_stream(
    executor: AgentExecutor,
    user_input: str,
    chat_history: list | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    流式运行 Agent，基于 astream_events(version="v2") 逐事件 yield。

    事件类型:
        {"type": "tool_start",  "name": "...", "input": "..."}
        {"type": "tool_end",    "name": "...", "output": "..."}
        {"type": "token",       "content": "..."}   # 最终答案的逐 token
        {"type": "done",        "answer": "...", "steps": [...]}
    """
    steps: list[dict[str, Any]] = []
    answer_parts: list[str] = []

    async for event in executor.astream_events(
        {"input": user_input, "chat_history": chat_history or []},
        version="v2",
    ):
        kind = event["event"]
        name = event.get("name", "")
        data = event.get("data", {})

        if kind == "on_tool_start":
            inp = data.get("input", "")
            yield {"type": "tool_start", "name": name, "input": str(inp)}
            steps.append({"tool": name, "tool_input": str(inp), "output": ""})

        elif kind == "on_tool_end":
            out = data.get("output", "")
            # 找到最后一个同名 tool_start 步骤填上输出
            for s in reversed(steps):
                if s["tool"] == name and s["output"] == "":
                    s["output"] = str(out)
                    break
            yield {"type": "tool_end", "name": name, "output": str(out)}

        elif kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            content = ""
            if chunk is not None:
                # AIMessageChunk.content 可能是 str 或 list
                c = getattr(chunk, "content", "")
                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    # tool_calls 的 content 可能是 list of dict
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in c
                    )
            # 仅 tool_calls 为空时的 content 才是答案 token
            tool_calls = getattr(chunk, "tool_call_chunks", None) if chunk else None
            if content and not tool_calls:
                answer_parts.append(content)
                yield {"type": "token", "content": content}

    yield {
        "type": "done",
        "answer": "".join(answer_parts),
        "steps": steps,
    }


def _extract_steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        if isinstance(action, AgentAction):
            steps.append({
                "tool": action.tool,
                "tool_input": str(action.tool_input),
                "output": str(observation),
            })
        elif isinstance(action, AgentFinish):
            steps.append({
                "tool": "__finish__",
                "tool_input": "",
                "output": str(action.return_values.get("output", "")),
            })
    return steps
