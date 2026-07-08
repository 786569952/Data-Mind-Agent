"""
data_platform/etl.py
=====================
数据清洗与预处理逻辑 (模拟数据平台 ETL 层)。

设计要点:
    每个清洗步骤都会产生一条结构化的 `CleanStep` 日志，前端可以据此
    可视化展示 "清洗前 / 清洗动作 / 清洗后" 的全过程，体现数据平台
    ETL 的透明可观测能力。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd


@dataclass
class CleanStep:
    """单步清洗记录，供前端可视化展示。"""

    step: str                  # 步骤名称
    description: str           # 检查/处理内容描述
    action: str                # 实际执行的动作
    affected: int              # 受影响行数
    before: dict[str, Any] = field(default_factory=dict)  # 处理前指标
    after: dict[str, Any] = field(default_factory=dict)   # 处理后指标

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "action": self.action,
            "affected": self.affected,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class ETLReport:
    """整轮 ETL 的汇总报告。"""

    original_rows: int
    cleaned_rows: int
    original_cols: int
    cleaned_cols: int
    steps: list[CleanStep] = field(default_factory=list)

    @property
    def removed_rows(self) -> int:
        return self.original_rows - self.cleaned_rows

    def to_summary(self) -> dict[str, Any]:
        return {
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "removed_rows": self.removed_rows,
            "original_cols": self.original_cols,
            "cleaned_cols": self.cleaned_cols,
            "n_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


def _shape(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}


def _missing(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def _duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def load_csv(path: str) -> pd.DataFrame:
    """读取 CSV (模拟 ODS 层原始数据落地)。"""
    return pd.read_csv(path)


def run_etl(df: pd.DataFrame, drop_duplicates: bool = True,
            fill_missing: bool = True, fix_dtypes: bool = True,
            fix_invalid_values: bool = True) -> tuple[pd.DataFrame, ETLReport]:
    """
    执行完整 ETL 流程并返回 (清洗后 DataFrame, ETL 报告)。

    每一步都会在 report.steps 中追加一条记录，前端可逐步渲染。
    """
    report = ETLReport(
        original_rows=int(df.shape[0]),
        cleaned_rows=int(df.shape[0]),
        original_cols=int(df.shape[1]),
        cleaned_cols=int(df.shape[1]),
    )

    def _record(step: str, desc: str, action: str, affected: int,
                before: dict, after: dict) -> None:
        report.steps.append(
            CleanStep(step=step, description=desc, action=action,
                      affected=affected, before=before, after=after)
        )

    # ---- Step 1: 去除完全重复行 ---------------------------------------
    if drop_duplicates:
        before_n = report.cleaned_rows
        before_dup = _duplicates(df)
        df = df.drop_duplicates().reset_index(drop=True)
        affected = before_n - int(df.shape[0])
        _record(
            step="去除重复行",
            desc="检测完全重复的记录行 (所有列均相同)。",
            action=f"删除 {affected} 条重复记录" if affected else "无重复记录，跳过",
            affected=affected,
            before={"rows": before_n, "duplicates": before_dup},
            after={"rows": int(df.shape[0]), "duplicates": _duplicates(df)},
        )
        report.cleaned_rows = int(df.shape[0])

    # ---- Step 2: 缺失值处理 ------------------------------------------
    if fill_missing:
        before_missing = _missing(df)
        before_rows = int(df.shape[0])
        actions: list[str] = []
        # 数值列用中位数填充
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            for c in num_cols:
                n = int(df[c].isna().sum())
                if n:
                    med = df[c].median()
                    df[c] = df[c].fillna(med)
                    actions.append(f"列 `{c}` 用中位数 {med} 填充 {n} 个缺失值")
        # 文本列用 "未知" 填充
        obj_cols = df.select_dtypes(exclude="number").columns.tolist()
        if obj_cols:
            for c in obj_cols:
                n = int(df[c].isna().sum())
                if n:
                    df[c] = df[c].fillna("未知")
                    actions.append(f"列 `{c}` 用 '未知' 填充 {n} 个缺失值")
        affected = before_missing - _missing(df)
        _record(
            step="缺失值处理",
            desc="数值列用中位数填充，文本列用 '未知' 填充。",
            action="; ".join(actions) if actions else "无缺失值，跳过",
            affected=affected,
            before={"missing_total": before_missing, "rows": before_rows},
            after={"missing_total": _missing(df), "rows": int(df.shape[0])},
        )

    # ---- Step 3: 数据类型修正 ----------------------------------------
    if fix_dtypes:
        before_dtypes = {c: str(df[c].dtype) for c in df.columns}
        fixed: list[str] = []
        # 日期列识别
        for c in df.columns:
            if "date" in c.lower() or "日期" in c:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    if str(before_dtypes[c]) != str(df[c].dtype):
                        fixed.append(f"`{c}` -> datetime64")
                except Exception:
                    pass
        # 数值列中混入字符串的修正
        for c in df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(df[c], errors="coerce")
            valid_rate = converted.notna().mean()
            # 仅当原列大部分可解析为数值时才转换 (避免误转 ID/编码列)
            if valid_rate > 0.8 and converted.notna().sum() > 0:
                # 保留原列名，但若大量丢失则回退
                if converted.isna().sum() <= df[c].isna().sum() + 1:
                    df[c] = converted
                    fixed.append(f"`{c}` -> numeric")
        _record(
            step="数据类型修正",
            desc="识别日期列并转换；将数值比例高的文本列转为数值类型。",
            action="; ".join(fixed) if fixed else "无需类型修正",
            affected=len(fixed),
            before={"dtypes": before_dtypes},
            after={"dtypes": {c: str(df[c].dtype) for c in df.columns}},
        )

    # ---- Step 4: 非法值修正 ----------------------------------------
    if fix_invalid_values:
        before_rows = int(df.shape[0])
        removed = 0
        fixed_actions: list[str] = []
        # 数量/价格为负 -> 视为非法，修正为绝对值并记录
        for c in df.columns:
            if df[c].dtype.kind in "iuf" and any(
                kw in c.lower() for kw in ("qty", "quantity", "数量", "price", "金额", "amount")
            ):
                neg = int((df[c] < 0).sum())
                if neg:
                    df[c] = df[c].abs()
                    fixed_actions.append(f"列 `{c}` 中 {neg} 个负值取绝对值修正")
        # 非法地区/分类 (例如 not_a_region) -> 标记为未知
        for c in ("region", "地区"):
            if c in df.columns:
                valid = {"华东", "华北", "华南", "西南", "西北", "东北", "华中", "未知"}
                bad = df[~df[c].isin(valid)]
                if len(bad):
                    df.loc[~df[c].isin(valid), c] = "未知"
                    fixed_actions.append(f"列 `{c}` 中 {len(bad)} 个非法值改为 '未知'")
        _record(
            step="非法值修正",
            desc="负向数值取绝对值；不在合法枚举内的地区/分类标记为 '未知'。",
            action="; ".join(fixed_actions) if fixed_actions else "未发现非法值",
            affected=len(fixed_actions),
            before={"rows": before_rows},
            after={"rows": int(df.shape[0])},
        )

    report.cleaned_rows = int(df.shape[0])
    report.cleaned_cols = int(df.shape[1])
    return df, report


def build_documents(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    将清洗后的 DataFrame 切片为 LangChain 可向量化的文档列表。
    每行一条记录，包含文本内容与元数据，模拟数据仓库的明细宽表。
    """
    docs: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        fields = [f"{col}: {row[col]}" for col in df.columns]
        docs.append({
            "page_content": " | ".join(fields),
            "metadata": {"row_index": int(idx), **{
                str(c): (str(row[c]) if pd.notna(row[c]) else "")
                for c in df.columns
            }},
        })
    return docs
