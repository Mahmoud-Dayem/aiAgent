from typing import Literal , TypedDict ,Annotated
from langgraph.graph import StateGraph , START, END
from langchain_ollama import ChatOllama
from pydantic import Field, BaseModel
from langgraph.graph.message import BaseMessage , add_messages
from langchain_core.messages import HumanMessage ,AIMessage
from langgraph.checkpoint.memory import MemorySaver
 
from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3
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
    
connection = sqlite3.connect(database="chatbot.db",check_same_thread=False)    
checkpoint = SqliteSaver(connection)
graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)

# define edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatbot = graph.compile(checkpoint)
 
 
 
def get_all_threads() -> list[str]:
    """Return saved thread IDs from SQLite, newest first."""
    seen = set()
    thread_ids = []

    for checkpoint_tuple in checkpoint.list(None):
        thread_id = checkpoint_tuple.config["configurable"].get("thread_id")

        if thread_id and thread_id not in seen:
            seen.add(thread_id)
            thread_ids.append(thread_id)

    return thread_ids


def get_thread_messages(thread_id: str) -> list[dict]:
    """Load one thread's complete chat history from SQLite."""
    config = {"configurable": {"thread_id": thread_id}}

    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])

    history = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        history.append(
            {
                "role": role,
                "content": message.content
            }
        )

    return history 


# memory


# config = {
#     "configurable": {
#         "thread_id": "user_1"
#     }
# }
# init={"messages": [HumanMessage(content="how are you brother")]}

# response = chatbot.invoke(init,config)
# print(response['messages'][-1].content)        