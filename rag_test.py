import os
# —— 1. 模型与 Embedding 导入 ——
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.documents import Document

# —— 2. LangChain 核心组件导入 ——
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# —— 3. 设置 API Key ——
os.environ["DASHSCOPE_API_KEY"] = "sk-31dc4cbb431940ee8fb7b6a0678f9e4f"

# —— 4. 初始化通义千问对话模型 ——
chat_model = ChatTongyi(model="qwen-plus-latest")

# —— 5. 加载并切分文档 ——
#    确保在项目根目录下建立一个 data/ 文件夹，并放入 test_doc.txt
#loader = TextLoader("./data/test_doc.txt", encoding="utf-8")
loader = DirectoryLoader("./data", glob="*.txt")
documents = loader.load()

#对文档进行初步清洗（去除所有空行）
cleaned = []
for doc in documents:
    text = doc.page_content
    # 去掉全是空白的行
    lines = [line for line in text.split("\n") if line.strip()]
    cleaned_text = "\n".join(lines)
    cleaned.append(Document(page_content=cleaned_text, metadata=doc.metadata))

documents = cleaned

#    拆分成多个段落：每 500 字为一块，重叠 50 字
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# —— 6. 向量化并建立 FAISS 索引 ——
emb_model = DashScopeEmbeddings(model="text-embedding-v1")
vector_store = FAISS.from_documents(docs, emb_model)

# —— 7. 构建 RetrievalQA Chain ——
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# —— 8. 交互式问答 ——
if __name__ == "__main__":
    # 先加入一个 System 消息，设定助手角色
    message_history = [
        SystemMessage(content="你是一个工业文档问答机器人，会参考已加载的文档给出简洁专业的回答。如果已加载的文档里没有用户问题的答案，你就直接用自身能力回答，不用参考已加载文档。")
    ]

    print("=== 文档问答机器人（RAG）启动 ===")
    print("（输入 exit 或 quit 可退出）\n")

    while True:
        user_input = input("👱 用户：")
        if user_input.lower() in ["exit", "quit"]:
            print("=== 会话结束 ===")
            break

        # 1) 将用户输入追加到历史
        message_history.append(HumanMessage(content=user_input))

        # 2) 调用 RetrievalQA：自动检索 Top-k 片段并调用 LLM 生成回答
        result = qa_chain.invoke({"query": user_input})
        answer = result["result"]
        message_history.append(AIMessage(content=answer))

        # 3) 打印回答
        print(f"\n🤖 助手：{answer}\n")

        # 4) 打印检索到的文档片段（可选，用于学习和调试）
        print("🗒️ 引用文档片段：")
        for idx, doc in enumerate(result["source_documents"], start=1):
            snippet = doc.page_content.strip().replace("\n", " ")[:200]
            print(f"  【段落{idx}】{snippet} ...")
        print("\n" + "=" * 60 + "\n")