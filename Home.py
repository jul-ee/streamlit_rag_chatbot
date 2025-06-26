import streamlit as st

st.set_page_config(page_title="RAG Chatbot Hub", page_icon="🤖")
st.title("💬 RAG Chatbot Demo Hub")
st.write(
    """
    왼쪽 사이드바에서 원하는 데모를 선택하세요.

    **페이지 목록**
    1. ChatBasic: 간단한 대화 제공
    2. ChatConstitution: 헌법 질의응답 제공
    3. ChatPDF: PDF 기반 대화 제공
    4. ChatConstitutionMem: 문맥을 파악하는 헌법 질의응답 제공
    """
)
