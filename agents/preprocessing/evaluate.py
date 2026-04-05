"""
전처리 품질 평가 (LLM judge).

3종 모델(gpt-4o-mini / claude-haiku / gemini-2.0-flash)로 샘플 200건을 전처리한 뒤
GPT-4o-mini를 judge로 사용해 품질을 비교하고 리포트를 저장한다.

사용 예시:
    python -m agents.preprocessing.evaluate
    → eval_report_preprocessing.json 저장
"""

import json
import logging
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

from agents.preprocessing.agent import preprocess_batch, sample_reviews


JUDGE_SYSTEM = """너는 텍스트 전처리 품질 평가 전문가야.
원문 리뷰와 전처리 결과를 비교해서 아래 3가지를 각 1~5점으로 평가해.

1. noise_removal: 반복 문자·이모지·불필요한 특수문자가 잘 제거됐는가
   5=완벽히 제거 / 3=일부 남음 / 1=거의 제거 안 됨

2. meaning_preserved: 원문의 핵심 내용과 감정이 유지됐는가
   5=완벽히 보존 / 3=일부 변형 / 1=핵심 손실

3. no_over_processing: 필요한 내용까지 삭제되지 않았는가
   5=과도한 처리 없음 / 3=일부 불필요하게 삭제 / 1=너무 많이 삭제됨

반드시 JSON 형식으로만 응답:
{"noise_removal": 점수, "meaning_preserved": 점수, "no_over_processing": 점수}"""


JUDGE_USER = """원문: {original}

전처리 결과: {preprocessed}"""


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

EVAL_MODELS = [
    "gpt-4o-mini",
    "claude-haiku-4-5-20251001",
    "gemini-2.0-flash",
]
JUDGE_MODEL = "gemini-2.0-flash-lite"
REPORT_PATH = "eval_report_preprocessing.json"

# 종합 점수 가중치
WEIGHTS = {
    "noise_removal": 0.4,
    "meaning_preserved": 0.4,
    "no_over_processing": 0.2,
}


def judge_llm(original: str, preprocessed: str, max_retries: int = 3) -> dict | None:
    """
    gemini-2.0-flash-lite로 전처리 결과 1건 채점.
    단순 비교·채점 작업이므로 저비용 모델로 충분.
    최대 max_retries회 재시도 후에도 실패하면 None 반환 → 집계에서 제외.
    """
    prompt = JUDGE_USER.format(original=original, preprocessed=preprocessed)
    model = genai.GenerativeModel(
        model_name=JUDGE_MODEL,
        system_instruction=JUDGE_SYSTEM,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0),
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.warning(f"Judge 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 2초, 4초 대기 후 재시도

    logger.error("Judge 최종 실패 — 스킵")
    return None


def run_evaluation(n_samples: int = 200) -> dict:
    """
    평가 전체 흐름:
    1. 층화 샘플링 n건
    2. 3종 모델로 각각 전처리
    3. LLM judge로 채점
    4. 모델별 평균 점수 집계
    """
    samples = sample_reviews(n_samples) # 층화추출

    # 모델별 전처리 (DB 저장 없이 메모리에만 보관)
    model_outputs: dict[str, list[dict]] = {}
    for model in EVAL_MODELS:
        logger.info(f"[{model}] 전처리 시작 ({len(samples)}건)")
        model_outputs[model] = preprocess_batch(samples, model) # 전처리

    # LLM judge 채점
    scores: dict[str, list[dict | None]] = {m: [] for m in EVAL_MODELS}
    total = len(samples)

    for i, sample in enumerate(samples):
        original = sample["review_text"]
        for model in EVAL_MODELS:
            preprocessed = model_outputs[model][i]["cleaned_text"]
            score = judge_one(original, preprocessed)
            scores[model].append(score)
            time.sleep(0.2)  # rate limit 대응

        if (i + 1) % 20 == 0:
            logger.info(f"Judge 진행: {i + 1}/{total}")

    # 모델별 평균 점수 집계 (None 제외)
    summary = {}
    for model in EVAL_MODELS:
        valid = [s for s in scores[model] if s is not None]
        failed = len(scores[model]) - len(valid)
        if failed:
            logger.warning(f"[{model}] Judge 스킵: {failed}건")

        n = len(valid)
        avg = {
            k: round(sum(s[k] for s in valid) / n, 3)
            for k in WEIGHTS
        }
        avg["total"] = round(sum(avg[k] * w for k, w in WEIGHTS.items()), 3)
        avg["failed_count"] = failed
        summary[model] = avg

    return {
        "n_samples": n_samples,
        "judge_model": JUDGE_MODEL,
        "weights": WEIGHTS,
        "summary": summary,
        "detail": scores,
    }

