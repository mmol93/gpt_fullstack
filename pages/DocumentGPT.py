import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

# message라는 키가 st.session_state 안에 있지 않을 때 실시
# 해당 조건이 없다면 매번 session_state에 있는 "message"를 초기화 해버린다.
if "messages" not in st.session_state:
    # message라는 키로 session_state을 만든다.
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

# session_state["messages"]에 있는 내용으로 메시지를 만든다.
for message in st.session_state["messages"]:
    # 메시지를 표시할 때는 저장할 필요가 없으므로 save=False 설정
    send_message(
        message["message"],
        message["role"],
        save=False,
    )


message = st.chat_input("Send a message to the ai ")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)