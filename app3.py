# uvicorn app:app --reload
from fastapi import FastAPI
from langserve import add_routes
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    "Explain this engineering topic: {topic}"
)

chain = prompt | llm | StrOutputParser()
print('hello world')

app = FastAPI()

add_routes(
    app,
    chain,
    path="/engineering"
)