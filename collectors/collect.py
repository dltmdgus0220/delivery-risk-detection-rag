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
