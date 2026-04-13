"""
리랭커 비교 평가 (MRR@5, NDCG@5, Latency P50/P95).

하이브리드 검색 top-20 → 각 리랭커 → top-5 결과를 비교.
Ground truth: 하이브리드 검색 결과에 대해 GPT-4o-mini judge로 관련성(0/1) 레이블링.
결정 기준: NDCG@5 차이 5% 미만이면 속도가 빠른 것 선택.

MRR 사용 이유:
  벡터 검색/BM25는 "관련 있는 것만" 뽑는 게 아니라 "유사도/키워드 점수 높은 순" 으로 뽑음.
  → top-20 안에 실제 관련 없는 문서가 섞임 (키워드만 겹치거나 의미상 가까운 무관 문서).
  → 리랭커가 관련 있는 것을 top-5 안에서 얼마나 앞으로 올렸는지 MRR로 측정 가능.

사용 예시:
    python -m rag.evaluate
    → eval_report_reranker.json 저장
"""

import json
import logging
import math
import os
import time

import numpy as np
from dotenv import load_dotenv

from agents.embedding.embedder import openai_client
from rag.reranker import SUPPORTED_RERANKERS, rerank
from rag.retriever import hybrid_search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_PATH = "eval_report_reranker.json"
EVAL_RERANKERS = ["cross-encoder", "cross-encoder-ko", "mmr"]  # cohere는 비용으로 기본 제외 (필요 시 추가)

EVAL_QUERIES: list[str] = [
    "배달이 너무 늦어요",
    "음식이 식어서 왔어요",
    "앱이 자꾸 오류가 나요",
    "배달비가 너무 비싸요",
    "고객센터가 불친절해요",
    "쿠폰 할인이 마음에 들어요",
    "앱 삭제하겠습니다",
    "다른 배달앱으로 갈아탈게요",
    "주문 취소가 안 돼요",
    "결제 오류가 발생해요",
    "포장이 엉망이에요",
    "음식 퀄리티가 좋아요",
    "배달원이 불친절했어요",
    "앱 UI가 불편해요",
    "환불 처리가 안 돼요",
    "리뷰 조작이 의심돼요",
    "배달 예상 시간이 너무 달라요",
    "최소 주문 금액이 너무 높아요",
    "적립금 사용이 복잡해요",
    "자주 쓰는 편리한 앱이에요",
]


# ── Ground truth 생성 (LLM judge) ──────────────────────────

JUDGE_SYSTEM = """너는 정보 검색 품질 평가 전문가야.
주어진 쿼리와 리뷰 청크를 보고, 청크가 쿼리에 답하는 데 관련이 있는지 판단해.

관련 있음(1): 청크 내용이 쿼리 주제와 직접적으로 연관된 경우
관련 없음(0): 청크 내용이 쿼리와 무관한 경우

반드시 JSON 형식으로만 응답: {"relevant": 0 또는 1}"""


def judge_relevance(query: str, chunk_text: str, max_retries: int = 3) -> int:
    """GPT-4o-mini로 (query, chunk) 관련성 판단. 실패 시 0 반환."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"쿼리: {query}\n\n청크: {chunk_text}"},
                ],
                temperature=0,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json")
            result = json.loads(raw.strip())
            return int(result.get("relevant", 0))
        except Exception as e:
            logger.warning(f"Judge 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return 0


# ── 평가 지표 ───────────────────────────────────────────────

def compute_mrr(ranked_chunks: list[dict], relevance: dict[int, int], k: int = 5) -> float:
    """MRR@k 계산. 첫 번째 관련 청크가 등장하는 순위의 역수."""
    for i, chunk in enumerate(ranked_chunks[:k], 1):
        if relevance.get(chunk["id"], 0) == 1:
            return 1.0 / i
    return 0.0


def compute_ndcg(ranked_chunks: list[dict], relevance: dict[int, int], k: int = 5) -> float:
    """NDCG@k 계산. relevance 키는 chunk id."""
    dcg = sum(
        relevance.get(c["id"], 0) / math.log2(i + 2)
        for i, c in enumerate(ranked_chunks[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── 평가 메인 ───────────────────────────────────────────────

def run_evaluation() -> dict:
    """
    평가 전체 흐름:
    1. 20개 쿼리별 하이브리드 검색 top-20
    2. GPT-4o-mini judge로 ground truth 생성
    3. 각 리랭커 적용 → top-5
    4. MRR@5 + NDCG@5 + Latency P50/P95 계산
    """
    # 1. 쿼리별 하이브리드 검색 + ground truth 생성
    query_candidates: list[list[dict]] = []
    ground_truth: list[dict[int, int]] = []  # chunk id → 관련성

    for q_idx, query in enumerate(EVAL_QUERIES):
        logger.info(f"[{q_idx+1}/{len(EVAL_QUERIES)}] 하이브리드 검색: '{query}'")
        candidates = hybrid_search(query, top_k=20)
        query_candidates.append(candidates)

        relevance: dict[int, int] = {}
        for chunk in candidates:
            relevance[chunk["id"]] = judge_relevance(query, chunk["chunk_text"])
            time.sleep(0.1)

        relevant_count = sum(relevance.values())
        logger.info(f"  → 후보 {len(candidates)}개 중 관련 {relevant_count}개")
        ground_truth.append(relevance)

    # 2. 리랭커별 평가
    summary: dict[str, dict] = {}
    for reranker_name in EVAL_RERANKERS:
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []

        for q_idx, query in enumerate(EVAL_QUERIES):
            candidates = query_candidates[q_idx]

            t0 = time.perf_counter()
            reranked = rerank(reranker_name, query, candidates, top_n=5)
            latency_ms = (time.perf_counter() - t0) * 1000

            mrrs.append(compute_mrr(reranked, ground_truth[q_idx], k=5))
            ndcgs.append(compute_ndcg(reranked, ground_truth[q_idx], k=5))
            latencies.append(latency_ms)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        summary[reranker_name] = {
            "mrr_at_5": round(float(np.mean(mrrs)), 4),
            "ndcg_at_5": round(float(np.mean(ndcgs)), 4),
            "latency_p50_ms": round(latencies_sorted[int(n * 0.5)], 1), # 중앙값
            "latency_p95_ms": round(latencies_sorted[int(n * 0.95)], 1), # 상위 5% 극단적 케이스의 지연 수준. 서비스 안정성을 판단하기 위함.
        }
        logger.info(
            f"[{reranker_name}] MRR@5={summary[reranker_name]['mrr_at_5']}, "
            f"NDCG@5={summary[reranker_name]['ndcg_at_5']}, "
            f"P50={summary[reranker_name]['latency_p50_ms']}ms, "
            f"P95={summary[reranker_name]['latency_p95_ms']}ms"
        )

    return {
        "n_queries": len(EVAL_QUERIES),
        "top_k_hybrid": 20,
        "top_n_rerank": 5,
        "rerankers_evaluated": EVAL_RERANKERS,
        "summary": summary,
        "queries": EVAL_QUERIES,
        "ground_truth": [{str(k): v for k, v in gt.items()} for gt in ground_truth],
    }


def save_report(report: dict, path: str = REPORT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"평가 리포트 저장: {path}")


def print_summary(summary: dict):
    print("\n=== 리랭커 비교 평가 결과 ===")
    print(f"{'리랭커':<20} {'MRR@5':>8} {'NDCG@5':>8} {'P50(ms)':>10} {'P95(ms)':>10}")
    print("-" * 62)
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["ndcg_at_5"]):
        print(f"{name:<20} {s['mrr_at_5']:>8.4f} {s['ndcg_at_5']:>8.4f} {s['latency_p50_ms']:>10.1f} {s['latency_p95_ms']:>10.1f}")

    best = max(summary, key=lambda m: summary[m]["ndcg_at_5"])
    scores = [summary[m]["ndcg_at_5"] for m in summary]
    diff = max(scores) - min(scores)

    print(f"\nNDCG@5 최고: {best} ({summary[best]['ndcg_at_5']})")
    if diff < 0.05:
        fastest = min(summary, key=lambda m: summary[m]["latency_p50_ms"])
        print(f"→ NDCG@5 차이 {diff:.4f} < 0.05 → 속도 우선: {fastest} 선택 권장")
    else:
        print(f"→ NDCG@5 차이 {diff:.4f} ≥ 0.05 → 성능 우선: {best} 선택 권장")

