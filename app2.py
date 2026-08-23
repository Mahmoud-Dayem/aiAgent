# uvicorn app:app --reload
from fastapi import FastAPI
from langserve import add_routes
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent 

llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0
)

# prompt = ChatPromptTemplate.from_template(
#     "Explain this engineering topic: {topic}"
# )

# chain = prompt | llm | StrOutputParser()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

agent = create_agent(
    model=llm,
    tools=[multiply],
    system_prompt="You are a helpful engineering assistant."
)

prompt = ChatPromptTemplate.from_template(
    "Explain this engineering topic: {topic}"
)

chain = prompt | agent | StrOutputParser()

app = FastAPI()

add_routes(
    app,
    chain,
    path="/engineering"
)
