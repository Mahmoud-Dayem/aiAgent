from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    reasoning=False,
    num_ctx=4096,
    num_predict=512,
    keep_alive="30m")

