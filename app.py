import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

# 👉 수정된 임포트 경로
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드 (OpenAI API 키 등)
load_dotenv()

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="PDF AI Q&A", page_icon="📄", layout="centered")
st.title("📄 PDF AI 질의응답 봇")
st.write("PDF 문서를 업로드하고 문서 내용에 대해 질문해 보세요!")

# --- 2. PDF 처리 및 벡터 스토어 생성 함수 (캐싱 적용) ---
@st.cache_resource(show_spinner=False)
def process_pdf(uploaded_file):
    # Streamlit에서 업로드된 파일을 PyMuPDF로 읽기
    pdf_bytes = uploaded_file.read()
    pdf_doc = fitz.open("pdf", pdf_bytes)

    text = ""
    for page in pdf_doc:
        text += page.get_text()
        text += '\n\n'

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)

    # 임베딩 및 FAISS 벡터 스토어 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

# --- 3. UI 컴포넌트: 파일 업로드 ---
uploaded_file = st.file_uploader("분석할 PDF 파일을 업로드해 주세요.", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("PDF 문서를 분석하고 임베딩하는 중입니다... (최초 1회만 실행됩니다)"):
        vectorstore = process_pdf(uploaded_file)
    st.success("✅ 문서 분석 완료! 이제 질문할 수 있습니다.")

    st.divider()

    # --- 4. UI 컴포넌트: 질문 입력 및 답변 생성 ---
    query = st.text_input("질문을 입력하세요:", value="암보험의 월 납입해야할 보험료가 얼마인지?")

    if st.button("답변 생성", type="primary"):
        if query.strip() == "":
            st.warning("질문을 입력해 주세요.")
        else:
            with st.spinner("답변을 생성하는 중입니다..."):
                # 유사도 검색
                docs = vectorstore.similarity_search_with_score(query)

                # 컨텍스트 추출
                context = ""
                for doc, score in docs:
                    context += doc.page_content + "\n\n"

                # LLM 설정 (최신 모델인 gpt-4o 사용)
                llm = ChatOpenAI(temperature=0, model='gpt-4o')

                # 프롬프트 설정
                prompt_template = """다음 배경 지식을 사용해서 질문에 대답해 주세요.

배경지식
{context}
============
질문
{question}"""

                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | llm | StrOutputParser()

                # 답변 생성
                inputs = {"context": context, "question": query}
                answer = chain.invoke(inputs)

                # 결과 출력
                st.subheader("🤖 AI의 답변")
                st.write(answer)

                # 원문 컨텍스트 보기
                with st.expander("참고한 원문 컨텍스트 보기"):
                    st.text(context)
else:
    st.info("먼저 좌측 상단의 'Browse files' 버튼을 눌러 PDF 파일을 업로드해 주세요.")
