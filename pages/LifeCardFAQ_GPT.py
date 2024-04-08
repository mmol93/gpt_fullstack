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

project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
faq_caching_file_path = os.path.join(project_root_path, '.cache/lifecard')
faq_caching_file_path_langchain = LocalFileStore("./.cache/lifecard/")
life_faq_sitemap_url = "https://lifecard.dga.jp/sitemap.xml"

llm = ChatOpenAI(
    temperature=0.1,
)

# Map Re-rank를 실시하는 prompt
answers_prompt = ChatPromptTemplate.from_template(
    """
ユーザーの質問に対する回答は、以下のコンテキストのみを使用して行ってください。分からない場合は、「わかりません」と答えるだけで、適当な回答はしないでください。次に、回答に0から5の範囲で点数をつけてください。回答がユーザーの質問に答えている場合は高い点数を、そうでない場合は低い点数をつけてください。回答の点数は必ず含めてください。たとえ0点でも構いません。
コンテキスト: {context}

例: 質問: 月はどのくらいの距離にありますか?

回答: 月は384,400kmの距離にあります。
点数: 5
質問: 太陽はどのくらいの距離にありますか?
回答: わかりません
点数: 0
あなたの番です! 

質問: {question}
"""
)


def parse_page(soup):
    target_soup = soup.find('div', class_="faq-box")
    print(target_soup.text)
    return (
        str(target_soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="WebページからFAQデータを更新中")
def load_website(url):
    st.write(faq_caching_file_path)
    cache_file_list = os.listdir(faq_caching_file_path)
    # 캐싱된 데이터가 없으면 웹에서 새로운 데이터를 로드한다.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    loader = SitemapLoader(url, parsing_function=parse_page)
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(OpenAIEmbeddings(), faq_caching_file_path_langchain)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})
    # if cache_file_list:
    #     # 캐싱된 데이터가 있는지 확인
    #     cached_embeddings = CacheBackedEmbeddings.from_bytes_store(OpenAIEmbeddings(), faq_caching_file_path_langchain)
    #     cached_docs = cached_embeddings.docs
    #     vector_store = FAISS.from_documents(cached_docs, cached_embeddings)
    #     return vector_store.as_retriever(3)
    # else:
        

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
    retriever = load_website(life_faq_sitemap_url)
    chain = ({
        "context": retriever,
        "question": RunnablePassthrough()
    }) | answers_prompt | llm

    result = chain.invoke(question)
    st.markdown(result.content)
