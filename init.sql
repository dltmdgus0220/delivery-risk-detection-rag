-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 원본 리뷰
CREATE TABLE IF NOT EXISTS raw_reviews (
    id SERIAL PRIMARY KEY,
    app_id VARCHAR(100),
    platform VARCHAR(50),
    reviewer_name VARCHAR(200),
    review_date DATE,
    rating INT,
    thumbs_up_count INT,
    review_text TEXT,
    app_version VARCHAR(50),
    collected_at TIMESTAMP DEFAULT NOW()
);

-- 중복 수집 방지 인덱스 (같은 앱+작성자+날짜+텍스트 조합)
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_reviews_unique
    ON raw_reviews (app_id, reviewer_name, review_date, review_text);
-- 인덱스를 만들어서 빠르게 찾을 수 있게 함. 하지만 조회용이라기 보다는 중복 방지용 인덱스.

-- 전처리된 리뷰
CREATE TABLE IF NOT EXISTS processed_reviews (
    id SERIAL PRIMARY KEY,
    raw_review_id INT REFERENCES raw_reviews(id) ON DELETE CASCADE,
    cleaned_text TEXT,
    processed_by VARCHAR(50),
    processed_at TIMESTAMP DEFAULT NOW()
);
-- 외래키. raw_reviews 테이블의 id를 참조. ON DELETE CASCADE는 raw_reviews 테이블의 데이터가 삭제되면 이 테이블의 데이터도 삭제됨.

-- 분류 결과 (멀티 라벨)
CREATE TABLE IF NOT EXISTS review_labels (
    id SERIAL PRIMARY KEY,
    raw_review_id INT REFERENCES raw_reviews(id) ON DELETE CASCADE,
    is_churn BOOLEAN DEFAULT FALSE,
    is_complaint BOOLEAN DEFAULT FALSE,
    is_suggestion BOOLEAN DEFAULT FALSE,
    is_positive BOOLEAN DEFAULT FALSE,
    classified_by VARCHAR(50),
    human_reviewed BOOLEAN DEFAULT FALSE,
    reviewed_at TIMESTAMP
);

-- 조회 성능을 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_raw_reviews_date ON raw_reviews (review_date); -- 날짜 조건 검색을 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_review_labels_reviewed ON review_labels (human_reviewed); -- 검토 조건 검색을 위한 인덱스
-- 인덱스가 많으면 쓰기 작업이 느려질 수 있지만 전체 데이터 중에 일부인 데이터를 찾기에는 좋음. 예를 들어 human_reviewed가 true인 데이터를 찾기에는 좋음.
