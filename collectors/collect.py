"""
수동 수집 CLI.

사용 예시:
    python -m collectors.collect                                       # 전날 리뷰 전체 → DB 저장
    python -m collectors.collect --date 2026-03-30                     # 특정 날짜 전체 → DB 저장
    python -m collectors.collect --start 2026-03-01 --end 2026-03-31
    python -m collectors.collect --date 2026-03-30 --count 100         # 최대 100건
    python -m collectors.collect --date 2026-03-30 --output csv        # CSV로 저장 (확인용)
"""
import argparse
import csv
import logging
import os
from datetime import date, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from google_play_scraper import reviews, Sort


load_dotenv()
logging.basicConfig(level=logging.INFO) # DEBUG < INFO < WARNING < ERROR < CRITICAL. 즉 DEBUG를 제외한 나머지 로그만 출력.
logger = logging.getLogger(__name__)

BAEMIN_APP_ID = "com.sampleapp"
PLATFORM = "google_play"
BATCH_SIZE = 200

engine = create_engine(os.environ["DATABASE_URL"]) # DB 연결 통로


INSERT_SQL = text("""
    INSERT INTO raw_reviews
        (app_id, platform, reviewer_name, review_date, rating, thumbs_up_count, review_text, app_version)
    VALUES
        (:app_id, :platform, :reviewer_name, :review_date, :rating, :thumbs_up_count, :review_text, :app_version)
    ON CONFLICT (app_id, reviewer_name, review_date, review_text) DO NOTHING
""")


def collect(app_id: str, platform: str, start_date: date, end_date: date, count: int = None) -> list[dict]:
    """
    Google Play에서 리뷰를 수집.
    google-play-scraper는 날짜 필터를 지원하지 않으므로
    최신순으로 배치 수집하며 start_date보다 오래된 리뷰가 나오면 중단한다.

    count=None(기본값): 기간 내 전체 수집
    count=N: 최대 N건으로 제한
    """
    filtered = []
    token = None

    while True:
        batch, token = reviews(
            app_id,
            lang="ko",
            country="kr",
            sort=Sort.NEWEST,
            count=BATCH_SIZE,
            continuation_token=token,
        )

        if not batch:
            break

        for r in batch:
            review_date = r["at"].date()

            if review_date > end_date:
                continue

            if review_date < start_date:
                return filtered

            filtered.append({
                "app_id": app_id,
                "platform": platform,
                "reviewer_name": r["userName"],
                "review_date": review_date,
                "rating": r["score"],
                "thumbs_up_count": r["thumbsUpCount"],
                "review_text": r["content"],
                "app_version": r.get("appVersion", None),
            })

            if count and len(filtered) >= count:
                return filtered

        if not token:
            break

    return filtered


def get_yesterday_range() -> tuple[date, date]:
    yesterday = date.today() - timedelta(days=1)
    return yesterday, yesterday


def save_to_db(reviews: list[dict]) -> int:
    saved = 0
    with engine.begin() as conn:
        for review in reviews:
            result = conn.execute(INSERT_SQL, review)
            saved += result.rowcount
    return saved


def save_to_csv(reviews: list[dict], path: str):
    if not reviews:
        logger.warning("저장할 데이터가 없습니다.")
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)

    logger.info(f"CSV 저장 완료: {path} ({len(reviews)}건)")


def parse_args():
    parser = argparse.ArgumentParser(description="배달의민족 리뷰 수동 수집")
    parser.add_argument("--date", type=date.fromisoformat, help="수집할 날짜 (YYYY-MM-DD)")
    parser.add_argument("--start", type=date.fromisoformat, help="수집 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="수집 종료일 (YYYY-MM-DD)")
    parser.add_argument("--count", type=int, default=None, help="최대 수집 건수 (기본값: 기간 내 전체)")
    parser.add_argument("--output", choices=["db", "csv"], default="db", help="저장 방식 (기본값: db)")
    parser.add_argument("--csv-path", default="reviews.csv", help="CSV 저장 경로 (기본값: reviews.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.date:
        start_date = end_date = args.date
    elif args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        start_date, end_date = get_yesterday_range()

    count_label = f"최대 {args.count}건" if args.count else "전체"
    logger.info(f"수집 범위: {start_date} ~ {end_date}, {count_label}")

    collected = collect(
        app_id=BAEMIN_APP_ID,
        platform=PLATFORM,
        start_date=start_date,
        end_date=end_date,
        count=args.count,
    )
    logger.info(f"{len(collected)}건 수집")

    if args.output == "csv":
        save_to_csv(collected, args.csv_path)
    else:
        saved = save_to_db(collected)
        logger.info(f"{saved}건 저장 (중복 제외)")


if __name__ == "__main__":
    main()
