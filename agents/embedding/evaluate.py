"""
임베딩 모델 비교 평가 (MRR@10, NDCG@10).

3종 모델로 샘플 리뷰를 임베딩한 뒤 미리 정의된 쿼리 20개를 사용해
검색 품질을 비교하고 리포트를 저장한다.

Ground truth 생성: 3종 모델 top-10 union에 대해 GPT-4o-mini judge로 관련성(0/1) 레이블링.
같은 ground truth를 모든 모델에 적용해 공정하게 비교.

사용 예시:
    python -m agents.embedding.evaluate
    → eval_report_embedding.json 저장
    → embedding_eval_results 테이블에 저장
"""

import json
import logging
import math
import os
import time

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from agents.embedding.chunker import chunk_review
from agents.embedding.embedder import SUPPORTED_MODELS, embed, openai_client

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

EVAL_MODELS = SUPPORTED_MODELS
REPORT_PATH = "eval_report_embedding.json"

# 한국어 쿼리 20개 (유형별: 배달/음식/앱/결제/CS)
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


# ── 샘플링 ─────────────────────────────────────────────────

def sample_and_chunk(n: int = 200) -> list[dict]:
    """processed_reviews에서 층화 샘플링 후 청킹."""
    per_rating = n // 5
    chunks: list[dict] = []

    with engine.connect() as conn:
        for rating in range(1, 6):
            rows = conn.execute(text("""
                SELECT p.raw_review_id, p.cleaned_text
                FROM processed_reviews p
                JOIN raw_reviews r ON p.raw_review_id = r.id
                WHERE r.rating = :rating
                ORDER BY RANDOM()
                LIMIT :limit
            """), {"rating": rating, "limit": per_rating}).fetchall()

            for row in rows:
                for idx, chunk_text in enumerate(chunk_review(row.cleaned_text)):
                    chunks.append({
                        "raw_review_id": row.raw_review_id,
                        "chunk_index": idx,
                        "chunk_text": chunk_text,
                    })

    logger.info(f"샘플링 완료: {n}건 → 청크 {len(chunks)}개")
    return chunks


# ── Ground truth 생성 (LLM judge) ─────────────────────────

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


# ── 검색 및 지표 계산 ──────────────────────────────────────

def cosine_search(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int = 10) -> list[int]:
    """정규화된 벡터 기준 내적 = 코사인 유사도로 top-k 인덱스 반환."""
    scores = doc_vecs @ query_vec
    return list(np.argsort(scores)[::-1][:top_k])


# 정답 리뷰가 처음 등장하는 순위의 역수 평균
# 얼마나 빨리 관련 있는 문서를 찾았는지
def compute_mrr(ranked_indices: list[int], relevant_set: set[int], k: int = 10) -> float:
    for i, idx in enumerate(ranked_indices[:k], 1):
        if idx in relevant_set:
            return 1.0 / i
    return 0.0


# NDCG는 관련도가 높은 문서가 얼마나 상위에 잘 배치됐는지를 이상적인 순서와 비교해 평가하는 지표로, 검색 결과의 전체 품질을 측정할 때 사용
def compute_ndcg(ranked_indices: list[int], relevance: dict[int, int], k: int = 10) -> float:
    dcg = sum(
        relevance.get(idx, 0) / math.log2(i + 2)
        for i, idx in enumerate(ranked_indices[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── 평가 메인 ──────────────────────────────────────────────

def run_evaluation(n_samples: int = 200) -> dict:
    """
    평가 전체 흐름:
    1. 층화 샘플링 + 청킹
    2. 3종 모델로 모든 청크 임베딩
    3. 20개 쿼리 임베딩 → 각 모델별 top-10 검색
    4. top-10 union → LLM judge ground truth 생성
    5. MRR@10, NDCG@10 계산
    6. embedding_eval_results 저장
    """
    chunks = sample_and_chunk(n_samples)
    chunk_texts = [c["chunk_text"] for c in chunks]

    # 1. 3종 모델 문서 임베딩
    doc_vecs: dict[str, np.ndarray] = {}
    for model in EVAL_MODELS:
        logger.info(f"[{model}] 문서 임베딩 시작 ({len(chunk_texts)}개 청크)")
        doc_vecs[model] = embed(model, chunk_texts, is_query=False)
        logger.info(f"[{model}] 완료: shape={doc_vecs[model].shape}")

    # 2. 쿼리 임베딩 + top-10 검색
    query_results: dict[str, list[list[int]]] = {m: [] for m in EVAL_MODELS}
    query_latencies: dict[str, list[float]] = {m: [] for m in EVAL_MODELS}

    for q in EVAL_QUERIES:
        for model in EVAL_MODELS:
            q_vec = embed(model, [q], is_query=True)[0]

            t0 = time.perf_counter()
            ranked = cosine_search(q_vec, doc_vecs[model], top_k=10)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            query_results[model].append(ranked)
            query_latencies[model].append(latency_ms)

    # 3. Ground truth 생성 (3종 top-10 union → LLM judge)
    logger.info("Ground truth 생성 시작 (LLM judge)")
    ground_truth: list[dict[int, int]] = []

    for q_idx, query in enumerate(EVAL_QUERIES):
        candidate_set: set[int] = set()
        for model in EVAL_MODELS: # 각 임베딩 모델로부터 나온 문서들을 모두 후보로 사용 (union)
            candidate_set.update(query_results[model][q_idx])

        relevance: dict[int, int] = {}
        for chunk_idx in candidate_set:
            relevance[chunk_idx] = judge_relevance(query, chunk_texts[chunk_idx])
            time.sleep(0.1)

        ground_truth.append(relevance)
        logger.info(
            f"쿼리 [{q_idx+1}/{len(EVAL_QUERIES)}] '{query[:20]}' "
            f"— 후보 {len(candidate_set)}개 중 관련 {sum(relevance.values())}개"
        )

    # 4. MRR@10, NDCG@10 계산
    summary: dict[str, dict] = {}
    for model in EVAL_MODELS:
        mrrs, ndcgs = [], []
        for q_idx in range(len(EVAL_QUERIES)):
            ranked = query_results[model][q_idx]
            rel_dict = ground_truth[q_idx]
            mrrs.append(compute_mrr(ranked, {idx for idx, r in rel_dict.items() if r == 1}))
            ndcgs.append(compute_ndcg(ranked, rel_dict))

        summary[model] = {
            "mrr_at_10": round(float(np.mean(mrrs)), 4),
            "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
            "latency_ms_avg": int(sum(query_latencies[model]) / len(query_latencies[model])),
        }

    _save_to_db(summary)

    return {
        "n_samples": n_samples,
        "n_chunks": len(chunk_texts),
        "n_queries": len(EVAL_QUERIES),
        "summary": summary,
        "queries": EVAL_QUERIES,
        "ground_truth": [{str(k): v for k, v in gt.items()} for gt in ground_truth],
    }

