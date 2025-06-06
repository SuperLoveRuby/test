from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

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