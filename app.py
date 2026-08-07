import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# =========================
# 1. 環境変数の読み込み (.env)
# =========================
here = os.path.dirname(__file__)
env_path = os.path.join(here, "01.env")
load_dotenv(env_path)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# =========================
# 2. RAGの準備（キャッシュして毎回実行されるのを防ぐ）
# =========================
@st.cache_resource
def setup_rag_chain():
    # PDFの読み込みと分割
    pdf_path = os.path.join(here, "sample.pdf") 
    if not os.path.exists(pdf_path):
        st.error(f"❌ {pdf_path} が見つかりません。同じフォルダにPDFファイルを配置してください。")
        st.stop()

    loader = PyPDFLoader(pdf_path)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # ベクトルDB (ChromaDB) に保存
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    db_path = os.path.join(here, "chroma_db")
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # AIモデルの準備 & RAGの組み立て
    chat_llm = ChatHuggingFace.from_model_id(
        "Qwen/Qwen2-1.5B-Instruct",
        task="conversational",
        backend="endpoint",
        provider="featherless-ai",
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_tokens=256,
    )

    system_prompt = (
        "あなたは優秀な資料分析アシスタントです。以下の参考資料（PDFから抽出したテキスト）のみを基に、ユーザーの質問に日本語で答えてください。\n"
        "もし参考資料に答えが含まれていない場合は、推測せず「資料には記載がありません」と正直に答えてください。\n\n"
        "【参考資料】\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chat_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# =========================
# 3. Streamlitの画面づくり
# =========================
st.title("MiroAI")

# RAGチェーンの準備（初回のみ時間がかかります）
with st.spinner("AIを準備中です...少しお待ちください"):
    rag_chain = setup_rag_chain()

# セッションにチャット履歴を保存するリストを用意
if "messages" not in st.session_state:
    st.session_state.messages = []

# 履歴をチャット形式で表示
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    with st.chat_message(role):
        st.write(msg.get("content", ""))

# ユーザーが文字を入力する欄を作る
user_input = st.chat_input("AIに質問してみよう！")

# 入力があれば履歴に追加してAIへ問い合わせ、応答を履歴に追加して表示
if user_input:
    # ユーザーメッセージを履歴に保存して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("AIが回答を作成中..."):
        try:
            # LangChainの処理を呼び出して回答をもらう
            response = rag_chain.invoke({"input": user_input})
            # レスポンスの取り出しに柔軟性を持たせる
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("output") or str(response)
            else:
                answer = str(response)

            # 応答を履歴に保存して表示
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")