import streamlit as st

st.title("AI Chatbot Test")

with st.container():
    # 임시 AI 비서 메시지 출력
    with st.chat_message("assistant"):
        st.write("저는 지금 테스트 중입니다. 질문을 입력하시면 임시 응답을 드립니다.")

if prompt := st.chat_input("메시지를 입력하세요..."):
    
    # 사용자 질문을 화면에 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    temp_response = f"**입력 확인 완료.** "
    
    # 임시 응답을 화면에 표시
    with st.chat_message("assistant"):
        st.markdown(temp_response)