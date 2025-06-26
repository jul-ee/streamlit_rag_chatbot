"""
streamlit_rag_upload.py
-----------------------
업로드된 PDF 1개를 대상으로 실시간 RAG 질의응답을 제공하는 RAG 챗봇

• secrets.toml에서 OPENAI_API_KEY 로드
• PDF → 페이지 분할(RecursiveCharacterTextSplitter) → OpenAI 임베딩
• Chroma 벡터스토어에 임베딩 저장(메모리 기반, 캐싱)
• LangChain RAG 체인(ChatPromptTemplate + GPT-4o)으로 답변 생성
• Streamlit chat UI + 세션 상태로 대화 기록 관리
"""

# ──────────────────────────────────────────────
# Ⅰ. Import & 환경 설정
# ──────────────────────────────────────────────
import os
import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Chroma 다중 테넌트 캐시 오류 방지
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

# secrets.toml → 런타임으로 API 키 로드
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="ChatPDF", page_icon="📄")

# ──────────────────────────────────────────────
# Ⅱ. 데이터 준비 / 벡터스토어
# ──────────────────────────────────────────────
@st.cache_resource
def load_pdf(_file):
    """UploadedFile ▶ 임시파일 ▶ LangChain Documents 반환"""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        loader = PyPDFLoader(file_path=tmp_file_path)
        pages = loader.load_and_split()
    return pages


@st.cache_resource
def create_vector_store(_docs):
    """문서 청크 → OpenAI 임베딩 → Chroma 벡터스토어"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
    )
    return vectorstore


def format_docs(docs):
    """검색된 Document → 하나의 문자열로 병합"""
    return "\n\n".join(doc.page_content for doc in docs)

# ──────────────────────────────────────────────
# Ⅲ. LangChain RAG 체인
# ──────────────────────────────────────────────
@st.cache_resource
def chaining(_pages):
    """업로드된 PDF 기반 RAG 체인 생성"""
    vectorstore = create_vector_store(_pages)
    retriever = vectorstore.as_retriever()

    qa_system_prompt = """
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer. \
    Please answer in Korean and use respectful language.\
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=OPENAI_API_KEY
    )

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ──────────────────────────────────────────────
# Ⅳ. Streamlit UI
# ──────────────────────────────────────────────
st.header("💬 ChatPDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    pages = load_pdf(uploaded_file)
    rag_chain = chaining(pages)

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "무엇이든 물어보세요!"}
        ]

    # 과거 대화 렌더링
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    # 사용자 질문 처리
    if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
        st.chat_message("human").write(prompt_message)
        st.session_state.messages.append({"role": "user", "content": prompt_message})

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.write(response)
