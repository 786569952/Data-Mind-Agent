"""
app.py
======
Data-Mind-Agent 可视化前端 (Streamlit)。

功能:
    1. 侧边栏配置模型 provider (阿里百炼 / Ollama llama)、对话模型、向量模型、
       temperature / top_p。
    2. 「数据摄入」页：上传或使用示例 CSV，逐步可视化 ETL 清洗过程。
    3. 「向量化」页：将清洗后的数据写入 ChromaDB，构建知识库。
    4. 「Agent 对话」页：与数据资产管理 Agent 交互，真流式输出 (打字机效果) +
       工具调用链实时展示。

启动:
    streamlit run app.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# 让项目根目录可被 import (agents / data_platform)
ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_platform import etl
from data_platform import document as doc_proc
from data_platform.embeddings import get_embeddings, get_ollama_embeddings, build_embeddings
from data_platform.vector_store import (
    build_vector_store, load_vector_store, reset_vector_store, docs_from_records,
)
from agents.rag_agent import build_agent, arun_agent_stream

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

SAMPLE_CSV = ROOT / "knowledge_base" / "sample_data.csv"

# ----------------------------- 页面基础设置 -----------------------------
st.set_page_config(
    page_title="Data-Mind-Agent | 数据资产管理 Agent",
    page_icon="🧠",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "etl_report": None,
        "vector_store": None,
        "chat_history": [],          # [{"role": "user"|"assistant", "content": "..."}]
        "agent_steps_history": [],   # 每轮的工具调用链
        # 非结构化文档处理 (与 CSV/Excel 互斥)
        "doc_text": None,            # 解析后的纯文本
        "doc_parse_info": None,      # ParseInfo
        "doc_chunks": None,          # list[ChunkInfo]
        "doc_embedding_report": None,  # EmbeddingReport
        "_embed_error": "",          # 向量化错误信息
        "_data_source": "",          # "structured" / "document"
        "_last_uploaded_name": "",   # 防重复处理：上次上传的文件名
        "_last_eval_result": None,   # RAGAS 最近一次评估结果
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ----------------------------- 侧边栏: 模型配置 -----------------------------
with st.sidebar:
    st.title("🧠 Data-Mind-Agent")
    st.caption("数据资产管理 Agent · 多模型驱动")

    st.markdown("### 🤖 模型 Provider")
    provider = st.radio(
        "选择大模型来源",
        options=["bailian", "ollama"],
        format_func=lambda x: "阿里百炼 (通义千问)" if x == "bailian" else "Ollama 本地 (llama)",
        horizontal=True,
        help="bailian: 阿里百炼 API；ollama: 本地运行的 llama / qwen 模型",
    )

    api_key = ""
    base_url = ""
    chat_model = "qwen-plus"

    if provider == "bailian":
        default_key = os.getenv("DASHSCOPE_API_KEY", "")
        api_key = st.text_input(
            "阿里百炼 API Key",
            value=default_key,
            type="password",
            help="在 https://bailian.console.aliyun.com 获取 API-KEY",
        )
        chat_model = st.selectbox(
            "对话模型",
            options=["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"],
            index=["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"].index(
                os.getenv("DEFAULT_CHAT_MODEL", "qwen-plus")
            ),
        )
    else:
        st.markdown("需先在本机运行 `ollama serve` 并 `ollama pull <模型>`")
        base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="默认本地 11434 端口",
        )
        chat_model = st.selectbox(
            "模型名称",
            options=["qwen2.5:7b", "llama3.1", "qwen2.5:3b", "llama3.2"],
            index=0,
            help="推荐 qwen2.5:7b (对 Agent 工具调用支持最好)；llama3.1 也可用但较慢",
        )
        st.caption("💡 推荐 `qwen2.5:7b`，工具调用最稳定。已检测模型见下方。")

    st.markdown("### 🎛️ 采样参数")
    c_t, c_p = st.columns(2)
    with c_t:
        temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05,
                                help="越高越随机，越低越确定")
    with c_p:
        top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05,
                          help="核采样：只从累计概率前 top_p 的 token 中采样")

    st.markdown("### 🧬 向量化 Provider")
    embed_provider = st.radio(
        "选择 Embedding 模型来源",
        options=["bailian", "ollama"],
        format_func=lambda x: "阿里百炼 (在线)" if x == "bailian" else "Ollama 本地 (离线)",
        horizontal=True,
        help="bailian: text-embedding-v1/v2/v3 (需 API Key)；ollama: 本地 bge-m3 等离线模型",
        key="embed_provider_radio",
    )

    embed_api_key = ""
    embed_model = "text-embedding-v3"

    if embed_provider == "bailian":
        embed_api_key = st.text_input(
            "百炼 API Key (向量化)",
            value=api_key or os.getenv("DASHSCOPE_API_KEY", ""),
            type="password",
            help="留空则复用上方对话模型的 API Key",
        )
        embed_model = st.selectbox(
            "向量模型 (百炼)",
            options=["text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
            index=2,
            help="v3 最新，1024 维；v2 兼容性好；v1 旧版",
        )
    else:
        st.caption("💡 推荐 `bge-m3`，中文效果好 (1024 维)")
        embed_model = st.selectbox(
            "向量模型 (Ollama)",
            options=["bge-m3", "nomic-embed-text", "mxbai-embed-large"],
            index=0,
            help="需先 `ollama pull <模型名>`。bge-m3 中文最佳",
        )
        # Ollama 复用对话页的 base_url
        if not base_url:
            base_url = "http://localhost:11434"

    st.divider()
    st.markdown("#### 📊 会话状态")
    st.write(f"- 结构化数据: {'✅' if st.session_state.cleaned_df is not None else '❌'}")
    st.write(f"- 文档文本:   {'✅' if st.session_state.doc_text is not None else '❌'}")
    st.write(f"- 切分块:     {len(st.session_state.doc_chunks) if st.session_state.doc_chunks else 0} 个")
    st.write(f"- 知识库:     {'✅' if st.session_state.vector_store is not None else '❌'}")

    if st.button("🗑️ 清空会话", use_container_width=True):
        for k in ["raw_df", "cleaned_df", "etl_report",
                  "vector_store", "doc_text", "doc_parse_info",
                  "doc_chunks", "doc_embedding_report"]:
            st.session_state[k] = None
        st.session_state["_embed_error"] = ""
        st.session_state["_data_source"] = ""
        st.session_state["_last_uploaded_name"] = ""
        for k in ("chat_history", "agent_steps_history"):
            st.session_state[k] = []
        st.rerun()


def chat_config_ready() -> bool:
    """对话模型是否已配置可用。"""
    if provider == "bailian":
        return bool(api_key and api_key.strip())
    return bool(chat_model.strip())


def embed_config_ready() -> bool:
    """向量化是否已配置可用。"""
    if embed_provider == "ollama":
        return bool(embed_model.strip())  # 本地模型，无需 key
    return bool(embed_api_key and embed_api_key.strip())


def _build_embed_client():
    """根据当前配置构建 Embedding 客户端 (统一入口)。"""
    return build_embeddings(
        provider=embed_provider,
        api_key=embed_api_key,
        model=embed_model,
        base_url=base_url or "http://localhost:11434",
    )


# ============================== 页面 Tabs ==============================
tab_ingest, tab_vector, tab_chat = st.tabs(
    ["📥 数据摄入", "🗄️ 向量化", "💬 Agent 对话"]
)


# ----------------------------- Tab 1: 数据摄入 -----------------------------
# 文件类型分组: 结构化 vs 非结构化文档
STRUCTURED_EXTS = ("csv", "xlsx", "xls")
DOC_EXTS = ("docx", "md", "txt", "prd")


with tab_ingest:
    st.header("📥 数据摄入与清洗 (ETL)")
    st.caption("支持结构化数据 (CSV/Excel → ETL) 和非结构化文档 (Word/Markdown/PRD/TXT → 解析切分)")

    c1, c2 = st.columns([3, 1])
    with c1:
        uploaded = st.file_uploader(
            "上传文件",
            type=list(STRUCTURED_EXTS + DOC_EXTS),
            help="CSV/Excel 走结构化清洗；Word/Markdown/PRD/TXT 走文档解析与切分",
        )
    with c2:
        st.markdown("&nbsp;")
        use_sample = st.button("📄 使用示例数据", use_container_width=True)

    # 判断当前数据源类型
    def _current_source() -> str:
        """返回 'structured' / 'document' / 'none'"""
        if st.session_state.cleaned_df is not None:
            return "structured"
        if st.session_state.doc_text is not None:
            return "document"
        return "none"

    if uploaded is not None:
        ext = uploaded.name.lower().rsplit(".", 1)[-1]
        # 防重复处理：同一个文件只在首次上传时解析，避免 rerun 覆盖切分/清洗结果
        if st.session_state.get("_last_uploaded_name") == uploaded.name:
            pass  # 已处理过，跳过
        elif ext in STRUCTURED_EXTS:
            # 切换到结构化数据：清掉文档支线
            for k in ("doc_text", "doc_parse_info", "doc_chunks", "doc_embedding_report",
                      "etl_report", "_embed_error"):
                st.session_state[k] = None if k != "_embed_error" else ""
            st.session_state.cleaned_df = None
            st.session_state.vector_store = None
            if ext == "csv":
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.raw_df = df
            st.session_state._data_source = "structured"
            st.session_state._last_uploaded_name = uploaded.name
            st.success(f"已上传结构化数据: {uploaded.name} ({df.shape[0]} 行 × {df.shape[1]} 列)")
        else:
            # 切换到非结构化文档：清掉结构化支线
            for k in ("raw_df", "cleaned_df", "etl_report",
                      "doc_chunks", "doc_embedding_report", "_embed_error"):
                st.session_state[k] = None if k != "_embed_error" else ""
            st.session_state.vector_store = None
            tmp_path = ROOT / "knowledge_base" / f"_upload_{uploaded.name}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            try:
                text, info = doc_proc.parse_document(str(tmp_path), uploaded.name)
                st.session_state.doc_text = text
                st.session_state.doc_parse_info = info
                st.session_state._data_source = "document"
                st.session_state._last_uploaded_name = uploaded.name
                st.success(
                    f"已解析文档: {uploaded.name} "
                    f"(类型={info.file_type}, 字符={info.total_chars}, 段落={info.n_paragraphs})"
                )
            except Exception as e:
                st.error(f"文档解析失败: {e}")
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
    elif use_sample:
        for k in ("doc_text", "doc_parse_info", "doc_chunks", "doc_embedding_report"):
            st.session_state[k] = None
        st.session_state.raw_df = pd.read_csv(SAMPLE_CSV)
        st.session_state._data_source = "structured"
        st.session_state._last_uploaded_name = ""
        st.success(f"已加载示例数据 ({st.session_state.raw_df.shape[0]} 行)")

    src = _current_source()

    # ============ 支线 A: 结构化数据 ETL ============
    if st.session_state.raw_df is not None:
        raw = st.session_state.raw_df
        st.subheader("1️⃣ 原始数据预览 (ODS 层)")
        st.dataframe(raw.head(20), use_container_width=True)

        with st.expander("🔍 原始数据问题概览", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总行数", raw.shape[0])
            m2.metric("缺失值总数", int(raw.isna().sum().sum()))
            m3.metric("重复行", int(raw.duplicated().sum()))
            m4.metric("列数", raw.shape[1])

        st.divider()
        st.subheader("2️⃣ 清洗配置")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            opt_dup = st.checkbox("去除重复行", value=True)
        with cc2:
            opt_miss = st.checkbox("缺失值处理", value=True)
        with cc3:
            opt_dtype = st.checkbox("类型修正", value=True)
        with cc4:
            opt_invalid = st.checkbox("非法值修正", value=True)

        if st.button("🚀 开始 ETL 清洗", type="primary", use_container_width=True):
            with st.spinner("正在执行 ETL..."):
                cleaned, report = etl.run_etl(
                    raw.copy(),
                    drop_duplicates=opt_dup,
                    fill_missing=opt_miss,
                    fix_dtypes=opt_dtype,
                    fix_invalid_values=opt_invalid,
                )
                st.session_state.cleaned_df = cleaned
                st.session_state.etl_report = report
            st.success("✅ 清洗完成！请查看下方清洗过程。")

    # ============ 支线 B: 非结构化文档解析 + 切分 → 结构化 ============
    if st.session_state.doc_text is not None and st.session_state.doc_parse_info is not None:
        info = st.session_state.doc_parse_info
        st.subheader("1️⃣ 文档解析结果 (ODS 层)")
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("文件类型", info.file_type)
        pm2.metric("总字符数", info.total_chars)
        pm3.metric("段落数", info.n_paragraphs)
        pm4.metric("表格数", info.n_tables)

        if info.headers:
            with st.expander(f"📑 文档标题结构 ({len(info.headers)} 个)", expanded=False):
                for h in info.headers:
                    st.text(h)

        with st.expander("📖 文档全文预览", expanded=False):
            st.text_area(
                "全文", value=st.session_state.doc_text, height=300,
                label_visibility="collapsed",
            )

        # ---- 切分参数 ----
        st.divider()
        st.subheader("2️⃣ 文档切分 (转为结构化数据)")

        with st.expander("📖 切分参数详解", expanded=False):
            st.markdown("""
**chunk_size (块大小)** — 每个文本块的最大字符数。越大上下文越完整但精度下降；越小检索越精准但易截断。中文推荐 300-800
**chunk_overlap (块重叠)** — 相邻块重叠字符数，通常为 chunk_size 的 10%-20%，防止句子被硬切断
**按 Markdown 标题切分** — 先按 #/##/### 分段，保留标题层级元数据
""")

        st.markdown("##### 🎯 推荐预设")
        presets = {
            "精确检索 (短块)": {"size": 300, "overlap": 50},
            "标准 (推荐)": {"size": 500, "overlap": 50},
            "保留上下文 (长块)": {"size": 1000, "overlap": 200},
            "成本优先 (超大块)": {"size": 1500, "overlap": 100},
        }
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 500
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 50

        preset_cols = st.columns(len(presets))
        for i, (name, p) in enumerate(presets.items()):
            with preset_cols[i]:
                if st.button(name, key=f"preset_{i}", use_container_width=True):
                    st.session_state.chunk_size = p["size"]
                    st.session_state.chunk_overlap = p["overlap"]
                    st.rerun()

        cp1, cp2 = st.columns(2)
        with cp1:
            chunk_size = st.slider(
                "chunk_size (块大小)", 100, 2000,
                st.session_state.chunk_size, 50,
                help="每个文本块的最大字符数",
            )
            st.session_state.chunk_size = chunk_size
        with cp2:
            chunk_overlap = st.slider(
                "chunk_overlap (块重叠)", 0, 500,
                st.session_state.chunk_overlap, 10,
                help="相邻块重叠字符数，通常为 chunk_size 的 10%-20%",
            )
            st.session_state.chunk_overlap = chunk_overlap

        is_md = info.file_type in ("md", "prd")
        use_md_headers = st.checkbox(
            "按 Markdown 标题切分", value=is_md, disabled=not is_md,
            help="先按 #/##/### 标题分段，再按字符递归切分",
        )

        if st.button("✂️ 执行切分 → 转为结构化数据", type="primary", use_container_width=True):
            with st.spinner("正在切分文档..."):
                try:
                    chunks = doc_proc.split_text(
                        st.session_state.doc_text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        is_markdown=use_md_headers,
                        source_filename=info.filename,
                    )
                    st.session_state.doc_chunks = chunks
                    # 切分块 → 结构化 DataFrame
                    rows = []
                    for c in chunks:
                        row = {
                            "chunk_index": c.index,
                            "page_content": c.text,
                            "char_len": c.char_len,
                            "source": c.metadata.get("source", ""),
                        }
                        for k in ("h1", "h2", "h3"):
                            if k in c.metadata:
                                row[k] = c.metadata[k]
                        rows.append(row)
                    st.session_state.cleaned_df = pd.DataFrame(rows)
                    st.session_state._data_source = "document"
                    st.success(f"✅ 切分完成，共 {len(chunks)} 块，已转为结构化数据 ({len(rows)} 行)。")
                except Exception as e:
                    st.error(f"切分失败: {e}")

        # 切分结果可视化
        if st.session_state.doc_chunks is not None:
            chunks = st.session_state.doc_chunks
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("块总数", len(chunks))
            cm2.metric("平均块长", int(sum(c.char_len for c in chunks) / max(len(chunks), 1)))
            cm3.metric("最长块", max((c.char_len for c in chunks), default=0))
            cm4.metric("最短块", min((c.char_len for c in chunks), default=0))

            st.bar_chart(
                pd.DataFrame({"chunk_index": [c.index for c in chunks],
                              "char_len": [c.char_len for c in chunks]}).set_index("chunk_index")
            )

            with st.expander(f"📋 查看切分块 ({len(chunks)} 个)", expanded=False):
                for c in chunks:
                    st.markdown(f"**Chunk {c.index}** · {c.char_len} 字符 · `{c.metadata}`")
                    st.text(c.text[:300] + ("..." if c.char_len > 300 else ""))
                    st.divider()

            st.subheader("3️⃣ 结构化数据预览 (DW 层)")
            st.dataframe(st.session_state.cleaned_df.head(20), use_container_width=True)

    # ============ 清洗过程可视化 (仅结构化) ============
    if st.session_state.etl_report is not None:
        report = st.session_state.etl_report
        st.divider()
        st.subheader("2️⃣ 清洗过程可视化")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("原始行数", report.original_rows)
        s2.metric("清洗后行数", report.cleaned_rows, delta=report.cleaned_rows - report.original_rows)
        s3.metric("清洗步骤数", len(report.steps))
        s4.metric("移除/修正记录", report.removed_rows)

        st.markdown("#### 🔬 逐步清洗详情")
        for i, step in enumerate(report.steps, 1):
            with st.container(border=True):
                head_col, badge = st.columns([4, 1])
                head_col.markdown(f"**Step {i} · {step.step}**")
                if step.affected > 0:
                    badge.markdown(f"`影响 {step.affected} 项`")
                else:
                    badge.markdown("`无变更`")

                st.caption(step.description)
                st.info(f"🛠️ 执行动作: {step.action}")

                b_col, a_col = st.columns(2)
                with b_col:
                    st.markdown("**清洗前**")
                    if step.before:
                        st.json(step.before)
                with a_col:
                    st.markdown("**清洗后**")
                    if step.after:
                        st.json(step.after)

        st.divider()
        st.subheader("3️⃣ 清洗后数据预览 (DW 层)")
        st.dataframe(st.session_state.cleaned_df.head(20), use_container_width=True)

        st.download_button(
            "⬇️ 下载清洗后数据 (CSV)",
            st.session_state.cleaned_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

    if src == "none" and uploaded is None and not use_sample:
        st.info("👆 请上传文件或使用示例数据。")


# ----------------------------- Tab 2: 向量化 -----------------------------
with tab_vector:
    st.header("🗄️ 向量化与知识库构建")
    st.caption("将结构化数据逐条向量化，写入 ChromaDB (DW 层)。文档类已切分为结构化行，统一处理。")

    if st.session_state.cleaned_df is None:
        st.warning("请先在「数据摄入」页上传文件并完成处理 (CSV 清洗 / 文档切分)。")
    elif not embed_config_ready():
        if embed_provider == "bailian":
            st.warning("请在侧边栏「向量化 Provider」填写阿里百炼 API Key，或切换为 Ollama 本地。")
        else:
            st.warning("请选择 Ollama 向量模型 (并确保本地 ollama serve 已启动)。")
    else:
        df = st.session_state.cleaned_df
        is_doc = "page_content" in df.columns
        src_label = "文档切分块" if is_doc else "结构化数据行"
        st.markdown(f"- 数据来源: **{src_label}**")
        st.markdown(f"- 待向量化记录数: **{len(df)}**")
        st.markdown(f"- 向量模型: `{embed_provider} / {embed_model}`")

        # 预览待向量化的文本内容
        with st.expander("📋 待向量化内容预览", expanded=False):
            if is_doc:
                for _, row in df.head(10).iterrows():
                    st.markdown(f"**记录 {row['chunk_index']}** · {row['char_len']} 字符")
                    st.text(row["page_content"][:300] + ("..." if row["char_len"] > 300 else ""))
                    st.divider()
            else:
                st.dataframe(df.head(10), use_container_width=True)

        st.divider()
        st.subheader("🚀 向量化执行")

        # ---- 构建向量库 (统一) ----
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🏗️ 构建向量库", type="primary", use_container_width=True):
                err_msg = ""
                try:
                    embeddings = _build_embed_client()
                    # 统一转为 ChunkInfo 列表
                    chunks = []
                    for idx, row in df.iterrows():
                        if is_doc:
                            text = str(row["page_content"])
                            meta = {"source": str(row.get("source", "")),
                                    "chunk_index": int(row.get("chunk_index", idx))}
                            for k in ("h1", "h2", "h3"):
                                if k in df.columns and pd.notna(row.get(k)):
                                    meta[k] = str(row[k])
                        else:
                            fields = [f"{c}: {row[c]}" for c in df.columns]
                            text = " | ".join(fields)
                            meta = {"row_index": int(idx)}
                        chunks.append(doc_proc.ChunkInfo(
                            index=int(idx), text=text,
                            char_len=len(text), metadata=meta,
                        ))
                    # 向量化 (带过程追踪)
                    docs, emb_report = doc_proc.embed_chunks(
                        chunks, embeddings,
                        model_name=f"{embed_provider}/{embed_model}",
                    )
                    st.session_state.doc_embedding_report = emb_report
                    # 写入 ChromaDB
                    reset_vector_store()
                    store = build_vector_store(docs, embeddings)
                    st.session_state.vector_store = store
                except Exception as e:
                    import traceback
                    err_msg = f"{e}\n```\n{traceback.format_exc()}\n```"
                    st.session_state.doc_embedding_report = None
                    st.session_state._embed_error = err_msg
                st.rerun()
        with col_b:
            if st.button("📂 加载已有向量库", use_container_width=True):
                try:
                    embeddings = _build_embed_client()
                    st.session_state.vector_store = load_vector_store(embeddings)
                    st.success("✅ 已加载持久化向量库。")
                except Exception as e:
                    st.error(f"加载失败: {e}")

        # 显示错误（若有）
        if st.session_state.get("_embed_error"):
            st.error(f"向量化失败:\n{st.session_state._embed_error}")
            if st.button("清除错误"):
                st.session_state._embed_error = ""
                st.rerun()

        # ---- Embedding 全过程可视化 ----
        if st.session_state.doc_embedding_report is not None:
            rep = st.session_state.doc_embedding_report
            st.success(
                f"✅ 向量化完成: 成功 {rep.success} / 失败 {rep.failed}，"
                f"向量维度 {rep.vector_dim}"
            )
            em1, em2, em3, em4, em5 = st.columns(5)
            em1.metric("模型", rep.model.split("/")[-1])
            em2.metric("总记录数", rep.total_chunks)
            em3.metric("成功", rep.success)
            em4.metric("失败", rep.failed)
            em5.metric("向量维度", rep.vector_dim)

            st.markdown("##### 🔬 逐条向量化详情")
            progress = st.progress(0.0, text="向量化进度")
            total = max(len(rep.steps), 1)
            for i, step in enumerate(rep.steps):
                with st.container(border=True):
                    sc1, sc2 = st.columns([3, 2])
                    with sc1:
                        badge = "✅" if step.status == "ok" else "❌"
                        st.markdown(f"{badge} **记录 {step.index}** · {step.char_len} 字符")
                        st.caption(f"文本预览: {step.text_preview}")
                        if step.error:
                            st.error(f"错误: {step.error}")
                    with sc2:
                        st.markdown(f"向量维度: `{step.vector_dim}`")
                        if step.vector_preview:
                            st.code(
                                "[" + ", ".join(f"{v:.6f}" for v in step.vector_preview) + ", ...]",
                                language="text",
                            )
                progress.progress((i + 1) / total,
                                  text=f"已向量化 {i + 1}/{total} 条")
            progress.empty()

        # ============ 检索测试 + RAGAS 评估 ============
        if st.session_state.vector_store is not None:
            st.divider()
            st.subheader("🔍 检索测试 & RAGAS 评估")
            st.caption("输入查询 → 检索 Top-K → (可选) 生成答案 → RAGAS 自动评分 (faithfulness / answer_relevancy / context_precision / context_recall)")

            test_q = st.text_input("输入查询，测试语义检索效果", key="ragas_test_q")
            col_k, col_gen, col_ref = st.columns([1, 2, 2])
            with col_k:
                top_k = st.slider("Top-K", 1, 10, 5, key="ragas_topk")
            with col_gen:
                do_generate = st.checkbox("生成答案 (用 Agent LLM)", value=True,
                                          help="检索后调用 LLM 生成答案，再评估")
            with col_ref:
                reference_answer = st.text_input(
                    "参考答案 (可选，用于 context_recall)",
                    help="提供 ground truth 才能计算 context_recall，否则跳过该指标"
                )

            if test_q:
                # ---- Step 1: 检索 ----
                with st.spinner(f"正在检索 Top-{top_k}..."):
                    try:
                        retrieved = st.session_state.vector_store.similarity_search(test_q, k=top_k)
                    except Exception as e:
                        st.error(f"检索失败: {e}")
                        retrieved = []

                if not retrieved:
                    st.warning("未检索到相关内容。")
                else:
                    st.markdown(f"##### 📄 检索结果 (Top-{len(retrieved)})")
                    contexts = []
                    for i, d in enumerate(retrieved, 1):
                        st.markdown(f"**[{i}]** {d.page_content[:300]}")
                        st.caption(f"metadata: {d.metadata}")
                        contexts.append(d.page_content)

                    # ---- Step 2: 生成答案 ----
                    answer = ""
                    if do_generate:
                        st.markdown("##### 🤖 生成答案")
                        try:
                            with st.spinner("正在用 Agent LLM 生成答案..."):
                                gen_llm = build_llm(
                                    provider, model_name, temperature, top_p, base_url
                                )
                                ctx_text = "\n\n".join(contexts[:3])
                                prompt = (
                                    f"请根据以下检索到的上下文回答问题。\n\n"
                                    f"上下文：\n{ctx_text}\n\n"
                                    f"问题：{test_q}\n\n"
                                    f"回答 (简洁准确)："
                                )
                                answer = gen_llm.invoke(prompt).content
                            st.info(answer)
                        except Exception as e:
                            st.error(f"生成答案失败: {e}")

                    # ---- Step 3: RAGAS 评估 ----
                    if do_generate and answer:
                        st.markdown("##### 📊 RAGAS 评估")
                        with st.expander("ℹ️ 指标说明", expanded=False):
                            st.markdown("""
- **faithfulness** (忠实度): 答案是否忠于检索上下文，无幻觉。1.0 = 完全忠于上下文
- **answer_relevancy** (答案相关性): 答案是否切题。1.0 = 完全切题
- **context_precision** (上下文精度): 检索结果中相关内容占比。1.0 = 全部相关
- **context_recall** (上下文召回): 参考答案中的信息是否被检索到。1.0 = 全部召回 (需提供参考答案)
""")
                        if st.button("🚀 开始 RAGAS 评分", type="primary", use_container_width=True):
                            try:
                                with st.spinner("正在用 LLM-as-Judge 评估 (可能需要 30-60 秒)..."):
                                    eval_llm = build_llm(
                                        provider, model_name, 0.0, 1.0, base_url
                                    )
                                    eval_emb = _build_embed_client()
                                    from agents.evaluator import evaluate_rag
                                    result = evaluate_rag(
                                        query=test_q,
                                        answer=answer,
                                        contexts=contexts,
                                        reference=reference_answer,
                                        llm=eval_llm,
                                        embeddings=eval_emb,
                                    )
                                    st.session_state._last_eval_result = result
                            except Exception as e:
                                st.error(f"评估失败: {e}")
                                st.session_state._last_eval_result = None

                        # 显示评估结果
                        if st.session_state.get("_last_eval_result"):
                            res = st.session_state._last_eval_result
                            st.success("✅ 评估完成")
                            valid_scores = {k: v for k, v in res.scores.items()
                                           if not k.endswith("_error") and v >= 0}
                            if valid_scores:
                                cols = st.columns(len(valid_scores))
                                for col, (k, v) in zip(cols, valid_scores.items()):
                                    col.metric(k, f"{v:.4f}")
                                    # 颜色提示
                                    if v >= 0.8:
                                        col.success("🟢 优秀")
                                    elif v >= 0.5:
                                        col.warning("🟡 一般")
                                    else:
                                        col.error("🔴 较差")

                            with st.expander("📋 详细结果", expanded=False):
                                st.json({
                                    "query": res.query,
                                    "answer": res.answer,
                                    "n_contexts": len(res.contexts),
                                    "reference": res.reference,
                                    "scores": res.scores,
                                })


# ----------------------------- Tab 3: Agent 对话 (流式) -----------------------------
def _consume_stream(executor, user_input, chat_history):
    """
    把 async generator (arun_agent_stream) 同步消费，逐事件 yield。
    Streamlit 的 st.write_stream 需要同步生成器。
    """
    agen = arun_agent_stream(executor, user_input, chat_history)
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                event = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            yield event
    finally:
        loop.close()


with tab_chat:
    st.header("💬 与数据资产管理 Agent 对话")
    st.caption("Agent 自主调用工具，回答以真流式 (打字机) 输出，工具调用过程实时展示")

    if not chat_config_ready():
        if provider == "bailian":
            st.warning("请在左侧侧边栏填写阿里百炼 API Key。")
        else:
            st.warning("请在左侧侧边栏填写 Ollama 模型名称 (并确保本地 ollama serve 已启动)。")
    else:
        status_md = [f"✅ 对话模型: {provider} / `{chat_model}`",
                     f"temperature={temperature}, top_p={top_p}"]
        if st.session_state.cleaned_df is not None:
            status_md.append("✅ 数据集已就绪")
        else:
            status_md.append("⚠️ 数据集未加载 (Agent 仅能回答通用问题)")
        if st.session_state.vector_store is not None:
            status_md.append("✅ 知识库已就绪")
        else:
            status_md.append("⚠️ 知识库未建立 (语义检索不可用)")
        st.markdown("  ".join(status_md))

        # 历史对话渲染
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("steps"):
                    with st.expander(f"🧩 工具调用链 ({len(msg['steps'])} 步)"):
                        for i, s in enumerate(msg["steps"], 1):
                            st.markdown(f"**Step {i} · `{s['tool']}`**")
                            if s.get("tool_input"):
                                st.caption(f"输入: {s['tool_input']}")
                            st.code(s.get("output", "")[:2000], language="text")

        if prompt := st.chat_input("向 Agent 提问，例如：哪个地区销售额最高？"):
            # 用户消息
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Agent 流式响应
            with st.chat_message("assistant"):
                ctx = {
                    "df": st.session_state.cleaned_df,
                    "vector_store": st.session_state.vector_store,
                }
                try:
                    executor = build_agent(
                        ctx, provider, chat_model,
                        api_key=api_key or None,
                        base_url=base_url or None,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except Exception as e:
                    st.error(f"Agent 构建失败: {e}")
                    st.stop()

                # 工具调用链容器 (实时填充)
                steps_holder = st.container()
                # 最终答案打字机容器
                answer_area = st.empty()
                streamed = []
                collected_steps = []

                with st.spinner("Agent 思考中..."):
                    try:
                        for event in _consume_stream(
                            executor, prompt, st.session_state.chat_history[:-1]
                        ):
                            etype = event.get("type")
                            if etype == "tool_start":
                                with steps_holder:
                                    st.info(f"🔧 调用工具 `{event['name']}` ...")
                                    with st.expander("输入", expanded=False):
                                        st.code(event.get("input", "")[:1000])
                            elif etype == "tool_end":
                                with steps_holder:
                                    st.success(f"✅ `{event['name']}` 返回结果:")
                                    with st.expander("输出", expanded=False):
                                        st.code(event.get("output", "")[:2000])
                                collected_steps.append({
                                    "tool": event["name"],
                                    "tool_input": "",
                                    "output": event.get("output", ""),
                                })
                            elif etype == "token":
                                streamed.append(event.get("content", ""))
                                answer_area.markdown("".join(streamed) + "▌")
                            elif etype == "done":
                                collected_steps = event.get("steps", collected_steps)
                                final_answer = event.get("answer") or "".join(streamed)
                                if not final_answer:
                                    final_answer = "(Agent 未返回文本答案，请查看上方工具调用结果)"
                                streamed = [final_answer]
                    except Exception as e:
                        streamed.append(f"\n\n⚠️ 流式出错: {e}")

                answer_area.markdown("".join(streamed))
                final_answer = "".join(streamed)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer,
                "steps": collected_steps,
            })
