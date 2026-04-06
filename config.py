"""
프로젝트 공통 설정.

파이프라인 동작 방식을 바꾸고 싶을 때 이 파일만 수정하면 된다.
비밀키·접속정보는 .env에서 관리.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# 청킹 방식
# "single" : 리뷰 전체를 1개 청크로 유지 (기본값)
# "aspect" : LLM으로 배달속도/음식품질/앱UX/가격/CS 단위로 분리
CHUNKER_MODE: str = os.getenv("CHUNKER_MODE", "single")

# 임베딩 모델
# "BAAI/bge-m3" | "intfloat/multilingual-e5-large" | "text-embedding-3-small"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
