from typing import Literal , TypedDict ,Annotated
from langgraph.graph import StateGraph , START, END
from langchain_ollama import ChatOllama
from pydantic import Field, BaseModel
from langgraph.graph.message import BaseMessage , add_messages
from langchain_core.messages import HumanMessage ,AIMessage
from langgraph.checkpoint.memory import MemorySaver
import  backend
import streamlit as st




config = {
    "configurable": {
        "thread_id": "user_1"
    }
}

st.title("dayem AI") 
user_input = st.text_input("Enter your message:")
init={"messages": [HumanMessage(content= user_input)]}
response = backend.chatbot.invoke(init,config)
print(response['messages'][-1].content) 


if user_input:
    with st.chat_message('user'):
        st.text(user_input)
    with st.chat_message('AI'):
        st.text(response['messages'][-1].content)
        
 