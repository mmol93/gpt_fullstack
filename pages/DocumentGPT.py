from typing import Dict, List
from uuid import UUID
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # *args는 일반적으로 받는 모든 인자를 의미
    # **kwargs는 key-value 형태로 받는 모든 인자를 의미(예: a=1, b=2...)
    def on_llm_start(self, *args, **kwargs):
        # ai가 말하는 것을 담기 위한 빈 상자 위젯을 만든다.
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # ai로 답변 작성이 끝났으면 답변을 저장한다.
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *arg, **kwargs):
        # 토큰을 받을 때마다 기존 메시지에 붙이도록 한다.
        self.message += token
        self.message_box.markdown(self.message)


# 함수에서 받은 file이 동일한 file이라면 해당 함수를 실행하지 않고 기존 반환값을 다시 반환한다.
@st.cache_data(show_spinner="Embedding files....")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    # 해당 위치에 파일을 저장한다.
    with open(file_path, "wb") as f:
        # (디버깅용)
        f.write(file_content)

    # 해당 위치에 캐싱한 데이터를 저장
    cache_dir = LocalFileStore(f"./.cache/embedings{file.name}")

    # chunk_size는 텍스트를 분할할 때 각 조각의 최대 길이를 지정한다.
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    text_loader = PyPDFLoader(f"./.cache/files/{file.name}")
    docs_from_text_loader = text_loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    # 임베디드한 데이터를 케싱처리
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # Chroma라는 Vector Store를 사용하여 저장되있는 임베디드 데이터를 사용한다.
    vectorstore = FAISS.from_documents(docs_from_text_loader, cached_embeddings)

    return vectorstore.as_retriever()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message=message, role=role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message=message["message"],
            role=message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chat bot with your files on the sidebar!

---

"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the following extracted parts of a long document and a question, create a final answer. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Context: {context}
""",
        ),
        ("human", "{question}"),
    ]
)

with st.sidebar:
    # file_uploader 위젯을 사용해서 파일 업로드를 받을 수 있게 한다.
    file = st.file_uploader("Upload a .txt or .pdf file.", type=["pdf", "txt"])

if file:
    retriever = embed_file(file=file)
    paint_history()
    message = st.chat_input("Ask anything!")
    if message:
        send_message(message=message, role="human")
        # chain에 필요한건 prompt에 넣어줄 값 | prompt | llm 임을 생각하고 만든다.
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            # chain.invoke를 chat_message 안에 넣어서 response가 chat_message로 나오게 한다.
            # 그렇게 하면 ChatCallbackHandler 안에 있는 것들이 똑같이 chat_message 안에서 실행되는것처럼 보인다.
            chain.invoke(message)

else:
    st.session_state["messages"] = []
