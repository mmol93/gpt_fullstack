from langchain.document_loaders import SitemapLoader
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
from langchain.chat_models import ChatOpenAI
import re
import json

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


def set_retrieval_data(input):
    docs = input["docs"]
    return {
        "question": input["question"],
        "answers": [
            {
                "answer": re.sub(r"\r+", " ", doc.page_content).strip(),
                "source": re.sub(r"\\n+", "", doc.metadata["source"]).strip(),
            }
            for doc in docs
        ],
    }


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


@st.cache_data(show_spinner="WebページからFAQデータを更新中")
def load_website(url):
    # st.write(faq_caching_web_cache_folder_path_pickle)
    cached_file = os.listdir(faq_caching_web_cache_folder_path_pickle)
    if cached_file:
        # 캐싱된 데이터가 있을 경우
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            OpenAIEmbeddings(), faq_caching_embedding_path
        )
        docs = open_loaded_data(faq_caching_file_path_pickle)
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        # 유사도 결과 상위 2개만 반환하도록 설정
        return vector_store.as_retriever(search_kwargs={"k": 4})
    else:
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
        return vector_store.as_retriever(search_kwargs={"k": 4})


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


def show_answer(input):
    question = input["question"]
    answers = json.dumps(input["answers"], ensure_ascii=False)
    
    answers_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                コンテキスト: {answers}

                以下のルールを絶対に守って返事してください。

                1. 上のコンテキストにある内容を元にし参考にしたLinkを参考として返事してください。具体的な例は以下の例を参考してください。
                2. コンテキストにあるデータはanswerとsourceになっている。必ずそれをPairで利用して欲しい。
                3. 返事は以下の例にあるMarkdown形式を必ず守ってください。
                4. 普通の挨拶は挨拶で返事するだけでいい。

                例：
                ## 複数回ログインに失敗し、ログインできない場合は一定時間空けてから再度お試しください。
                ## 解決方法
                1. 少しい時間がたったあともう一度試してください。
                2. そうしてもログインできない場合はチャットや電話お問い合わせを利用してください。
                ## 参考
                * https://lifecard.dga.jp/faq_detail.html?id=2676
                """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = answers_prompt | llm
    return chain.invoke({"question": question, "answers": answers})


if question:
    paint_history()
    send_message(message=question, role="human")
    retriever = load_website(life_faq_sitemap_url)
    chain = (
        ({"docs": retriever, "question": RunnablePassthrough()})
        | RunnableLambda(set_retrieval_data)
        | RunnableLambda(show_answer)
    )

    with st.chat_message("ai"):
        chain.invoke(question)
else:
    st.session_state["messages"] = []
