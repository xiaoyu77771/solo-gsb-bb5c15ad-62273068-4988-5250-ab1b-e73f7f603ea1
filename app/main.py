"""抄本核对 API:对三份独立抄录做三维全局对齐并生成共识文本。"""
from __future__ import annotations

import unicodedata

import grapheme
from fastapi import FastAPI, HTTPException

from .reconcile import AdjudicationError, adjudicate, reconcile, reconcile_anchored
from .schemas import (
    AdjudicateRequest,
    AdjudicateResponse,
    AnchoredReconcileRequest,
    AnchoredReconcileResponse,
    ReconcileRequest,
    ReconcileResponse,
)

MAX_GRAPHEMES = 120

app = FastAPI(
    title="抄本核对 API",
    version="1.2.0",
    description="接收三段抄录文本,NFC 规范化后做三维全局动态规划对齐,输出共识文本、对齐列与总成本;支持锚点强制对齐与人工裁决。",
)


def _to_clusters(texts: list[str]) -> list[list[str]]:
    """NFC 规范化并切分为字素簇序列;超过上限时抛 422。"""
    seqs: list[list[str]] = []
    for idx, raw in enumerate(texts):
        normalized = unicodedata.normalize("NFC", raw)
        clusters = list(grapheme.graphemes(normalized))
        if len(clusters) > MAX_GRAPHEMES:
            raise HTTPException(
                status_code=422,
                detail=f"texts[{idx}] 超过 {MAX_GRAPHEMES} 个字素簇(当前 {len(clusters)} 个)",
            )
        seqs.append(clusters)
    return seqs


def _validate_anchors(
    anchors: list, seqs: list[list[str]]
) -> list[tuple[int, int, int]]:
    """校验锚点不越界且各来源位置严格递增,返回三元组列表;违反时抛 422。"""
    validated: list[tuple[int, int, int]] = []
    prev = (-1, -1, -1)
    for idx, anchor in enumerate(anchors):
        pos = tuple(anchor.positions)
        for s in range(3):
            if not 0 <= pos[s] < len(seqs[s]):
                raise HTTPException(
                    status_code=422,
                    detail=f"anchors[{idx}].positions[{s}]={pos[s]} 越界:texts[{s}] 共 {len(seqs[s])} 个字素簇",
                )
            if pos[s] <= prev[s]:
                raise HTTPException(
                    status_code=422,
                    detail=f"anchors[{idx}].positions[{s}]={pos[s]} 未严格递增(上一锚点为 {prev[s]}),锚点不得重复或回退",
                )
        validated.append(pos)
        prev = pos
    return validated


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reconcile", response_model=ReconcileResponse)
def post_reconcile(req: ReconcileRequest) -> dict:
    return reconcile(*_to_clusters(req.texts))


@app.post("/reconcile/anchored", response_model=AnchoredReconcileResponse)
def post_reconcile_anchored(req: AnchoredReconcileRequest) -> dict:
    seqs = _to_clusters(req.texts)
    return reconcile_anchored(*seqs, _validate_anchors(req.anchors, seqs))


@app.post("/reconcile/adjudicate", response_model=AdjudicateResponse)
def post_reconcile_adjudicate(req: AdjudicateRequest) -> dict:
    seqs = _to_clusters(req.texts)
    anchors = _validate_anchors(req.anchors, seqs)
    decisions = [d.model_dump() for d in req.decisions]
    try:
        return adjudicate(*seqs, anchors, decisions)
    except AdjudicationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
