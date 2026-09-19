"""请求/响应模型。"""
from __future__ import annotations

import unicodedata

import grapheme
from pydantic import BaseModel, Field, field_validator, model_validator


class ReconcileRequest(BaseModel):
    """恰好三段待核对文本。"""

    texts: list[str] = Field(..., description="恰好三段文本,每段不超过 120 个字素簇")

    @field_validator("texts")
    @classmethod
    def _exactly_three(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError("必须恰好提供三段文本(texts 长度须为 3)")
        return v


class Anchor(BaseModel):
    """一个锚点:三份文本中必须同列对齐的字素簇位置(0 起始)。"""

    positions: list[int] = Field(
        ..., description="恰好三个位置,依次为 texts[0]/texts[1]/texts[2] 中的字素簇下标"
    )

    @field_validator("positions")
    @classmethod
    def _exactly_three(cls, v: list[int]) -> list[int]:
        if len(v) != 3:
            raise ValueError("锚点必须恰好给出三个位置(positions 长度须为 3)")
        return v


class AnchoredReconcileRequest(ReconcileRequest):
    """三段文本加锚点列表。"""

    anchors: list[Anchor] = Field(
        default_factory=list,
        description="锚点列表;各来源位置须严格递增、不越界、不重复,否则返回 422",
    )


class SourceCell(BaseModel):
    """某列中一个来源的字素及其原位置;空位均为 null。"""

    grapheme: str | None
    position: int | None


class Column(BaseModel):
    index: int = Field(..., description="列序号(0 起始)")
    mask: int = Field(..., description="本列消费的来源位掩码,bit0/1/2 对应 texts[0/1/2]")
    cost: int = Field(..., description="本列成本")
    sources: list[SourceCell] = Field(..., description="三个来源的字素与原位置")
    consensus: str = Field(..., description="本列共识字素,未达成时为 �")


class UncertainColumn(BaseModel):
    column: int = Field(..., description="未达成共识的列序号")
    sources: list[SourceCell] = Field(..., description="三个来源的字素与原位置")


class ReconcileResponse(BaseModel):
    consensus: str = Field(..., description="共识文本")
    total_cost: int = Field(..., description="对齐总成本")
    columns: list[Column] = Field(..., description="对齐列")
    uncertain: list[UncertainColumn] = Field(..., description="写入 � 的列及各来源字素与原位置")


class AnchoredColumn(Column):
    """附带 anchored 标记的对齐列。"""

    anchored: bool = Field(..., description="本列是否为锚点强制对齐列")


class AnchoredReconcileResponse(ReconcileResponse):
    columns: list[AnchoredColumn] = Field(..., description="对齐列(含 anchored 标记)")


class Decision(BaseModel):
    """一项人工裁决:以坐标签名定位一个 uncertain 列,并指定采用值。

    定位坐标为该列三个来源字素的原位置三元组(空位为 null);采用值
    二选一:source 选中该列某个非空来源字素,replacement 直接给出
    恰好一个字素簇(服务端 NFC 规范化后校验)。
    """

    positions: list[int | None] = Field(
        ...,
        description="恰好三个原位置(0 起始),依次为 texts[0/1/2] 中的字素簇下标,空位为 null",
    )
    source: int | None = Field(
        default=None, description="采用该列某个非空来源的字素:0/1/2 对应 texts[0/1/2]"
    )
    replacement: str | None = Field(
        default=None, description="替换字素,NFC 规范化后须为恰好一个字素簇"
    )

    @field_validator("positions")
    @classmethod
    def _exactly_three(cls, v: list[int | None]) -> list[int | None]:
        if len(v) != 3:
            raise ValueError("坐标签名必须恰好给出三个位置(positions 长度须为 3)")
        return v

    @field_validator("source")
    @classmethod
    def _valid_source(cls, v: int | None) -> int | None:
        if v is not None and v not in (0, 1, 2):
            raise ValueError("source 只能是 0、1 或 2")
        return v

    @field_validator("replacement")
    @classmethod
    def _single_cluster(cls, v: str | None) -> str | None:
        if v is None:
            return v
        normalized = unicodedata.normalize("NFC", v)
        clusters = list(grapheme.graphemes(normalized))
        if len(clusters) != 1:
            raise ValueError("replacement 须为恰好一个字素簇(NFC 规范化后)")
        return normalized

    @model_validator(mode="after")
    def _exactly_one_value(self) -> "Decision":
        if (self.source is None) == (self.replacement is None):
            raise ValueError("source 与 replacement 必须且只能提供一个")
        return self


class AdjudicateRequest(AnchoredReconcileRequest):
    """三段文本、可选锚点,加人工裁决列表。"""

    decisions: list[Decision] = Field(
        default_factory=list,
        description="人工裁决列表;全部有效才一次性应用,任一项失效返回 409",
    )


class AuditEntry(BaseModel):
    """一条裁决审计记录。"""

    signature: list[int | None] = Field(..., description="定位列的坐标签名(三个原位置,空位为 null)")
    column: int = Field(..., description="裁决命中的列序号")
    original: str = Field(..., description="裁决前列的共识字素(恒为 �)")
    adopted: str = Field(..., description="裁决采用的字素")
    method: str = Field(..., description="裁决方式:source(采用来源字素)或 replacement(替换字素)")
    source: int | None = Field(default=None, description="method 为 source 时采用的来源编号,否则为 null")


class AdjudicateResponse(BaseModel):
    consensus: str = Field(..., description="裁决后的完整文本")
    uncertain: list[UncertainColumn] = Field(..., description="仍未裁决的列及各来源字素与原位置")
    audit: list[AuditEntry] = Field(..., description="已应用裁决的审计项,顺序与请求 decisions 一致")
