from typing import Literal , TypedDict ,Annotated
from langgraph.graph import StateGraph , START, END
from langchain_ollama import ChatOllama
from pydantic import Field, BaseModel
from langgraph.graph.message import BaseMessage , add_messages
from langchain_core.messages import HumanMessage ,AIMessage
from langgraph.checkpoint.memory import MemorySaver
model = "qwen3.5:9b"
 

llm = ChatOllama(
    model=model,
    temperature=0
)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState)->ChatState:
    message = state['messages']
   
    response = llm.invoke(message)
    
    return{
        "messages":[response]
    }
checkpoint = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)

# define edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatbot = graph.compile(checkpoint)

# memory



# init={"messages": [HumanMessage(content="My name is user 2")]}
# response = workflow.invoke(init,config)
# print(response['messages'][-1].content)        