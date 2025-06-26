"""
streamlit_chat.py
-----------------
단일 LLM(ChatGPT) 기반 채팅 인터페이스를 제공하는 RAG 챗봇

• secrets.toml에서 OPENAI_API_KEY 로드
• 환경 변수 대신 st.secrets 로부터 OPENAI_API_KEY 읽기
• 세션 상태(st.session_state)를 활용하여 대화 이력을 유지
• Streamlit-native chat UI(st.chat_message / st.chat_input) 사용
"""

# ──────────────────────────────────────────────
# Ⅰ. Import & 환경 설정
# ──────────────────────────────────────────────
import os                      # (예비) OS 관련 유틸
import streamlit as st         # Streamlit UI 프레임워크
from langchain_openai import ChatOpenAI  # OpenAI LLM 래퍼

# secrets.toml → 런타임으로 API 키 로드
OPENAI_API_KEY: str = st.secrets["OPENAI_API_KEY"]


# ──────────────────────────────────────────────
# Ⅱ. LLM 초기화
# ──────────────────────────────────────────────
chat = ChatOpenAI(
    model="gpt-4o",            # 사용 모델 (mini/터보 등으로 변경 가능)
    temperature=0,             # 창의성 제어   (0 = deterministic)
    openai_api_key=OPENAI_API_KEY
)


# ──────────────────────────────────────────────
# Ⅲ. Streamlit 화면 구성
# ──────────────────────────────────────────────
st.title("💬 Chatbot")


# 1) 첫 방문 시 기본 인사말 세팅
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# 2) 저장된 대화 이력을 화면에 렌더링
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# 3) 사용자 입력 처리
if prompt := st.chat_input():                       # 사용자가 메시지를 입력하면
    # 3-1. 세션 상태에 사용자 메시지 저장 & 출력
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    # 3-2. LLM 호출 → 답변 생성
    response = chat.invoke(prompt)
    answer: str = response.content

    # 3-3. 세션 상태에 AI 답변 저장 & 출력
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.chat_message("assistant").write(answer)
