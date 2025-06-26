"""
streamlit_rag_memory.py
-----------------------
헌법 PDF를 기반으로 대화 문맥을 반영한 RAG 챗봇을 제공하는 RAG 챗봇

• secrets.toml에서 OPENAI_API_KEY 로드
• Chroma 벡터 DB + LangChain History-Aware Retriever 사용
• GPT 모델 선택(selectbox) → 실시간 체인 생성
• StreamlitChatMessageHistory로 대화 기록 ↔ LangChain 히스토리 연동
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
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# secrets.toml → 런타임으로 키 읽기
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="ChatConstitutionMem", page_icon="💬")


# ──────────────────────────────────────────────
# Ⅱ. PDF 로드 & 벡터스토어
# ──────────────────────────────────────────────
@st.cache_resource
def load_and_split_pdf(file_path):
    """PDF 파일 → LangChain Document 리스트"""
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


@st.cache_resource
def create_vector_store(_docs):
    """문서 청크 → 임베딩 → Chroma 저장"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        ),
        persist_directory=persist_directory
    )
    return vectorstore


@st.cache_resource
def get_vectorstore(_docs):
    """기존 Chroma DB가 있으면 로드, 없으면 생성"""
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=OPENAI_API_KEY
            )
        )
    else:
        return create_vector_store(_docs)


def format_docs(docs):
    """검색된 Document 들을 하나로 합치기"""
    return "\n\n".join(doc.page_content for doc in docs)


# ──────────────────────────────────────────────
# Ⅲ. LangChain 컴포넌트 초기화
# ──────────────────────────────────────────────
@st.cache_resource
def initialize_components(selected_model):
    """PDF 로드 → 히스토리-어웨어 RAG 체인 구성"""
    file_path = r"./data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 1) 이전 대화를 보고 질문을 재구성하는 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 2) 실제 Q&A에 사용할 시스템 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer perfect. please use imogi with the answer. \
대답은 한국어로 하고, 존댓말을 써줘.\

{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 3) LLM 설정
    llm = ChatOpenAI(
        model=selected_model,
        openai_api_key=OPENAI_API_KEY
    )

    # 4) History-Aware Retriever & 최종 RAG 체인
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# ──────────────────────────────────────────────
# Ⅳ. Streamlit UI
# ──────────────────────────────────────────────
st.header("💬 대화가 이어지는 헌법 Q&A 챗봇")

option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(option)

# LangChain ↔ Streamlit 히스토리 연결
chat_history = StreamlitChatMessageHistory(key="chat_messages")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 대화 히스토리 초기 메시지
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "헌법에 대해 무엇이든 물어보세요!"
    }]

# 기존 대화 내용 출력
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 사용자 질문 처리
if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config
            )
            answer = response['answer']
            st.write(answer)

            # 참고 문서 펼치기
            with st.expander("참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
