"""
streamlit_rag_local.py
--------------------------------------
대한민국 헌법 PDF를 벡터 DB(Chroma)에 임베딩한 뒤
GPT-4o-mini 모델로 질의응답을 수행하는 RAG 챗봇

• secrets.toml에서 OPENAI_API_KEY 로드
• 캐싱(@st.cache_resource)으로 PDF 파싱·벡터화 비용 절감
• Chroma 벡터스토어가 있으면 재사용, 없으면 새로 생성
• Streamlit chat UI + 세션 상태로 대화 기록 유지
"""

# ──────────────────────────────────────────────
# Ⅰ. Import & 환경 설정
# ──────────────────────────────────────────────
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# secrets.toml → 런타임으로 API 키 로드
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# ──────────────────────────────────────────────
# Ⅱ. PDF 로드 & 벡터스토어
# ──────────────────────────────────────────────
@st.cache_resource
def load_and_split_pdf(file_path):
    """PDF 경로 → LangChain Document 리스트"""
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


@st.cache_resource
def create_vector_store(_docs):
    """
    문서를 1 000자 단위로 분할한 뒤
    OpenAI 임베딩 → Chroma 벡터 DB에 저장
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)

    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        ),
        persist_directory="./chroma_db"
    )
    return vectorstore


@st.cache_resource
def get_vectorstore(_docs):
    """기존 Chroma DB가 있으면 로드, 없으면 새로 생성"""
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=OPENAI_API_KEY
            )
        )
    return create_vector_store(_docs)


def format_docs(docs):
    """검색된 Document 들을 하나의 문자열로 합침"""
    return "\n\n".join(doc.page_content for doc in docs)


# ──────────────────────────────────────────────
# Ⅲ. LangChain RAG 체인
# ──────────────────────────────────────────────
@st.cache_resource
def chaining():
    """PDF 로드 → 벡터스토어 → RAG 체인 조립"""
    file_path = r"./data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    qa_system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer perfect. Please use emoji with the answer.
    Please answer in Korean and use respectful language.
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), ("human", "{input}")]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY    # API 키 전달
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
st.header("헌법 Q&A 챗봇 💬 📚")
rag_chain = chaining()

# 1) 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}
    ]

# 2) 과거 대화 렌더링
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# 3) 사용자 입력 처리
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
