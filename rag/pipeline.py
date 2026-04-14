"""
End-to-End RAG 파이프라인.

쿼리 → 하이브리드 검색(top-20) → 리랭킹(top-5) → 결과 반환.

사용 예시:
    python -m rag.pipeline --query "배달이 너무 늦어요"
    python -m rag.pipeline --query "앱 오류가 자꾸 나요" --reranker mmr
"""

import argparse
import logging

from dotenv import load_dotenv

from rag.reranker import SUPPORTED_RERANKERS, rerank
from rag.retriever import hybrid_search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder"


def run_pipeline(query: str, reranker_name: str = DEFAULT_RERANKER, top_n: int = 5) -> list[dict]:
    """
    RAG 파이프라인 메인 함수.

    1. 하이브리드 검색 (벡터 + BM25 + RRF) → top-20
    2. 리랭킹 → top-5
    3. 결과 반환

    Args:
        query         : 자연어 검색 쿼리
        reranker_name : 리랭커 종류 (기본: cross-encoder)
        top_n         : 최종 반환 청크 수 (기본 5)

    Returns:
        청크 dict 리스트 (id, raw_review_id, chunk_index, chunk_text 포함)
    """
    logger.info(f"쿼리: '{query}' | 리랭커: {reranker_name}")

    candidates = hybrid_search(query, top_k=20)
    logger.info(f"하이브리드 검색 완료: {len(candidates)}개 후보")

    results = rerank(reranker_name, query, candidates, top_n=top_n)
    logger.info(f"리랭킹 완료: 최종 {len(results)}개 반환")

    return results


def print_results(query: str, results: list[dict]):
    print(f"\n=== 검색 결과: '{query}' ===\n")
    for i, chunk in enumerate(results, 1):
        print(f"[{i}] chunk_id={chunk['id']} | review_id={chunk['raw_review_id']}")
        print(f"    {chunk['chunk_text'][:200]}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 파이프라인 — 쿼리 검색 테스트")
    parser.add_argument("--query", required=True, help="검색 쿼리")
    parser.add_argument(
        "--reranker",
        choices=SUPPORTED_RERANKERS,
        default=DEFAULT_RERANKER,
        help=f"리랭커 종류 (기본값: {DEFAULT_RERANKER})",
    )
    parser.add_argument("--top-n", type=int, default=5, help="최종 반환 청크 수 (기본값: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(args.query, args.reranker, args.top_n)
    print_results(args.query, results)


if __name__ == "__main__":
    main()
