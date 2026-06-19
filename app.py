import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 1. 환경 변수 로드 (API KEY 등)
load_dotenv()

# 2. Streamlit 페이지 설정
st.set_page_config(page_title="오타쿠 AI 비서", page_icon="🤖")
st.title("🤖 오타쿠 AI 비서입니다능!")
st.write("무엇이든 물어보라능! (주인님을 위해 최선을 다하겠다능!)")

# 3. 사용자 입력 받기
user_input = st.text_input("질문을 입력하세요:", placeholder="예: 안녕!")

# 4. '전송' 버튼이 눌렸을 때의 동작
if st.button("전송"):
    if user_input.strip(): # 입력값이 비어있지 않은 경우에만 실행
        with st.spinner("답변을 작성하고 있다능... 기다려달라능!"):
            try:
                # LLM 모델 초기화 (기존 코드의 모델명 유지)
                llm = ChatOpenAI(model="gpt-5.5") 

                # 프롬프트 설정
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "너는 유능한 AI비서야. 일본 오타쿠 스타일로 대답해줘."),
                    ("user", "{input}")
                ])

                # Chain 구성
                chain = prompt | llm | StrOutputParser()

                # 결과 도출
                result = chain.invoke({"input": user_input})

                # 화면에 결과 출력
                st.success("도착했다능!")
                st.info(result)

            except Exception as e:
                st.error(f"에러가 발생했다능... 삐리릿: {e}")
    else:
        st.warning("질문을 먼저 입력해 달라능!")
