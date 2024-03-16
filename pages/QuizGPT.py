import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt3.5-turbo-1106"
)

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

# 함수에서 받은 file이 동일한 file이라면 해당 함수를 실행하지 않고 기존 반환값을 다시 반환한다.
@st.cache_data(show_spinner="Loading files....")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

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

    return docs_from_text_loader

docs = None
with st.sidebar:
    choice = st.selectbox(
        "Choose What you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader("Upload a pdf file", type=["pdf"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Name of the article")
        if topic: 
            # top_k_results: 위키피디아 결과에서 나온 결과 중 top 몇 개의 데이터만 가져올지
            retriever = WikipediaRetriever(top_k_results=3, lang="ko")
            with st.status("Searching from Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
Welecome to QuizGPT.

Choose Quiz from Wikipedia or your file.
"""
    )

else:
    st.write(docs)