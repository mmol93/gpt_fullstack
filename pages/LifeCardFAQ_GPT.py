from typing import Dict, List
from uuid import UUID
from langchain.document_loaders import SitemapLoader
from langchain.schema.output import LLMResult
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import os
from langchain.storage import LocalFileStore
import pickle
from langchain.callbacks.base import BaseCallbackHandler

project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
faq_caching_embedding_check_path = os.path.join(project_root_path, ".cache/lifecard")
faq_caching_embedding_path = LocalFileStore("./.cache/lifecard/")
faq_caching_web_cache_folder_path_pickle = os.path.join(
    project_root_path, ".cache/lifecard_faq_web"
)
faq_caching_file_path_pickle = os.path.join(
    project_root_path, ".cache/lifecard_faq_web/lifecard_faq_web.pkl"
)
life_faq_sitemap_url = "https://lifecard.dga.jp/sitemap.xml"


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # *args는 일반적으로 받는 모든 인자를 의미
    # **kwargs는 key-value 형태로 받는 모든 인자를 의미(예: a=1, b=2...)
    def on_llm_start(self, *args, **kargs):
        # ai가 말하는 것을 담기 위한 빈 상자 위젯을 만든다.
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kargs):
        # ai로 답변 작성이 끝났으면 답변을 저장한다.
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *arg, **kwargs):
        # 토큰을 받을 때마다 기존 메시지에 붙이도록 한다.
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message=message, role=role)


# 이때까지 입력한 채팅 내역을 출력
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message=message["message"],
            role=message["role"],
            save=False,
        )


llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

# Map Re-rank를 실시하는 prompt
answers_prompt = ChatPromptTemplate.from_template(
    """
ユーザーの質問に対する回答は、以下のコンテキストのみを使用して行ってください。分からない場合は、「わかりません」と答えるだけで、適当な回答はしないでください。
また、参考にしたをのLinkを一緒に出力してください。
コンテキスト: {context}

例: 質問: 月はどのくらいの距離にありますか?

回答: 月は384,400kmの距離にあります。
\nLink: https://www.exampl.com

質問: 太陽はどのくらいの距離にありますか?
回答: わかりません

あなたの番です! 

質問: {question}
"""
)


def save_loaded_data(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def open_loaded_data(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def parse_page(soup):
    target_soup = soup.find("div", class_="faq-box")
    print(target_soup.text)
    return (
        str(target_soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Webページ。からFAQデータを更新中")
def load_website(url):
    # st.write(faq_caching_web_cache_folder_path_pickle)
    cached_file = os.listdir(faq_caching_web_cache_folder_path_pickle)
    if cached_file:
        print("위에 실행됨")
        # 캐싱된 데이터가 있을 경우
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            OpenAIEmbeddings(), faq_caching_embedding_path
        )
        docs = open_loaded_data(faq_caching_file_path_pickle)
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        print("위에 작업 끝")
        # 유사도 결과 상위 2개만 반환하도록 설정
        return vector_store.as_retriever(search_kwargs={"k": 2})
    else:
        print("밑에 실행됨")
        # 캐싱된 데이터가 없을 경우
        # 캐싱된 데이터가 없으면 웹에서 새로운 데이터를 로드한다.
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        loader = SitemapLoader(url, parsing_function=parse_page)
        loader.requests_per_second = 2
        docs = loader.load_and_split(text_splitter=splitter)
        save_loaded_data(docs, faq_caching_file_path_pickle)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            OpenAIEmbeddings(), faq_caching_embedding_path
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        print("밑에 작업 끝")
        return vector_store.as_retriever(search_kwargs={"k": 2})


st.set_page_config(
    page_title="LifeCardFAQ GPT",
    page_icon="💳",
)


st.markdown(
    """
    # LifeCardFAQ GPT
            
    ようこそ「よくある質問」のGPTへ
"""
)
question = st.chat_input(
    "質問内容を入力してください。",
)

if question:
    paint_history()
    send_message(message=question, role="human")
    retriever = load_website(life_faq_sitemap_url)
    chain = (
        ({"context": retriever, "question": RunnablePassthrough()})
        | answers_prompt
        | llm
    )

    with st.chat_message("ai"):
        chain.invoke(question)
else:
    st.session_state["messages"] = []
