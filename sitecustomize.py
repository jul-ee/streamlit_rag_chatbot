"""
sitecustomize.py  —  Python 이 시작될 때 자동 실행
1) 최신 SQLite(pysqlite3)로 교체
2) Chroma 0.4.x 의 root 제한 우회
"""

import sys, os

# ① 최신 sqlite3 로 덮어쓰기
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError as err:
    # 설치가 안 됐으면 바로 눈에 보이도록
    print("pysqlite3-binary NOT installed:", err, file=sys.stderr)

# ② Chroma root-check 우회(0.4.22 ~ 0.4.24)
os.environ.setdefault("ALLOW_CHROMA_ROOT", "true")
