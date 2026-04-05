"""
분류 품질 평가 (LLM judge).

3종 모델(gpt-4o-mini / claude-haiku-4-5-20251001 / gemini-2.5-flash)로 샘플 200건을 분류한 뒤
gemini-2.5-flash-lite를 judge로 사용해 품질을 비교하고 리포트를 저장한다.

사용 예시:
    python -m agents.classification.evaluate
    → eval_report_classification.json 저장
"""

import json
import logging
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

from agents.classification.run import classify_batch, sample_reviews

JUDGE_SYSTEM = """너는 텍스트 분류 품질 평가 전문가야.
배달앱 리뷰와 분류 결과를 보고 아래 3가지를 각 1~5점으로 평가해.

[분류 구조]
- label: 리뷰의 주요 의도 (churn=이탈 의사 / complaint=불만 / positive=긍정)
- is_suggestion: 개선 제안 포함 여부 (label과 독립적)

[평가 기준]
1. label_accuracy: label이 리뷰의 주요 의도와 일치하는가
   5=정확 / 3=애매하지만 납득 가능 / 1=명백히 틀림

2. suggestion_accuracy: is_suggestion이 올바르게 판단됐는가
   5=정확 / 3=애매 / 1=명백히 틀림

3. overall_quality: 전체적으로 분류 결과가 리뷰를 잘 표현하는가
   5=매우 적절 / 3=보통 / 1=부적절

반드시 JSON 형식으로만 응답:
{"label_accuracy": 점수, "suggestion_accuracy": 점수, "overall_quality": 점수}"""

JUDGE_USER = """리뷰: {review}

분류 결과: {labels}"""

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

EVAL_MODELS = [
    "gpt-4o-mini",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
]
JUDGE_MODEL = "gemini-2.5-flash-lite"
REPORT_PATH = "eval_report_classification.json"

WEIGHTS = {
    "label_accuracy": 0.5,
    "suggestion_accuracy": 0.2,
    "overall_quality": 0.3,
}


def judge_one(review: str, labels: dict, max_retries: int = 3) -> dict | None:
    """
    gemini-2.5-flash-lite로 분류 결과 1건 채점.
    최대 max_retries회 재시도 후에도 실패하면 None 반환 → 집계에서 제외.
    """
    prompt = JUDGE_USER.format(review=review, labels=json.dumps(labels, ensure_ascii=False))
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
            candidate = response.candidates[0]
            if candidate.finish_reason != 1:
                raise ValueError(f"finish_reason={candidate.finish_reason}")
            text = candidate.content.parts[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json")
            return json.loads(text.strip())
        except Exception as e:
            logger.warning(f"Judge 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error("Judge 최종 실패 — 스킵")
    return None


def run_evaluation(n_samples: int = 200) -> dict:
    """
    평가 전체 흐름:
    1. 층화 샘플링 n건
    2. 3종 모델로 각각 분류
    3. LLM judge로 채점
    4. 모델별 평균 점수 집계
    """
    samples = sample_reviews(n_samples)

    model_outputs: dict[str, list[dict]] = {}
    for model in EVAL_MODELS:
        logger.info(f"[{model}] 분류 시작 ({len(samples)}건)")
        model_outputs[model] = classify_batch(samples, model)

    scores: dict[str, list[dict | None]] = {m: [] for m in EVAL_MODELS}
    total = len(samples)

    for i, sample in enumerate(samples):
        review = sample["cleaned_text"]
        for model in EVAL_MODELS:
            labels = model_outputs[model][i]["labels"]
            score = judge_one(review, labels)
            scores[model].append(score)
            time.sleep(0.2)

        if (i + 1) % 20 == 0:
            logger.info(f"Judge 진행: {i + 1}/{total}")

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


def save_report(report: dict, path: str = REPORT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"평가 리포트 저장: {path}")


def print_summary(summary: dict):
    print("\n=== 분류 모델 비교 평가 결과 ===")
    print(f"{'모델':<35} {'라벨 정확도':>12} {'제안 정확도':>12} {'전체 품질':>10} {'종합':>8}")
    print("-" * 82)
    for model, s in sorted(summary.items(), key=lambda x: -x[1]["total"]):
        print(
            f"{model:<35} "
            f"{s['label_accuracy']:>12.3f} "
            f"{s['suggestion_accuracy']:>12.3f} "
            f"{s['overall_quality']:>10.3f} "
            f"{s['total']:>8.3f}"
        )
    best = max(summary, key=lambda m: summary[m]["total"])
    print(f"\n최적 모델: {best}  (종합 점수: {summary[best]['total']})")
    print(f"→ 전체 분류 실행: python -m agents.classification.run --model {best}")


def main():
    report = run_evaluation(n_samples=200)
    save_report(report)
    print_summary(report["summary"])


if __name__ == "__main__":
    main()
