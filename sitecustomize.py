"""
sitecustomize.py
———————
* 모든 Python 프로세스가 시작될 때 자동 실행
* pysqlite3 → sqlite3 모듈 치환해서 Chroma가 최신 SQLite(3.45+)를 사용하도록 함
"""

import sys, os

# Chroma 0.4.x root 체크 우회 (필요 시)
os.environ.setdefault("ALLOW_CHROMA_ROOT", "true")

# sqlite3 패치
try:
    import pysqlite3                               # 최신 SQLite 내장
    sys.modules["sqlite3"] = pysqlite3             # 전역 교체
except ImportError:
    # pysqlite3-binary 가 설치되지 않은 환경에서는 패스
    pass
