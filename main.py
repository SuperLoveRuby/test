import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 设置阿里云 DashScope 的 API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-31dc4cbb431940ee8fb7b6a0678f9e4f"
# 设置调用的模型版本号
model = ChatTongyi(model="qwen-plus-latest")
#response = model.invoke([HumanMessage(content="你好，请介绍一下你自己")])
#print("🤖 回复：", response.content)

def get_completion_from_messages(messages, temperature=0.3):
    response = model.invoke(
        input = messages,
        temperature = temperature)
    return response.content

messages = [HumanMessage(content="上海工业数字化研究院是什么单位？")]

'''
messages = [
    SystemMessage(content="你是一个精通工业运维的智能助手，请用简洁专业的语言回答问题。"),
    HumanMessage(content="什么是空压机？"),
    AIMessage(content="空压机是将空气压缩以提高压力的设备，常用于制造业、电子、汽车等行业。"),
    HumanMessage(content="那它的运行温度一般是多少？"),
]
'''
reply = get_completion_from_messages(messages)
print("🤖 回答：", reply)