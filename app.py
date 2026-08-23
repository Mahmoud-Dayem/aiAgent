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

st.title("Dayem AI") 

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input("Enter your message:")

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.write(user_input)
        
    init={"messages": [HumanMessage(content= user_input)]}
        
    response = backend.chatbot.invoke(init,config) 
    ai_response = response['messages'][-1].content         
    st.session_state['message_history'].append({'role':'assistant','content': ai_response})
        
    with st.chat_message('assistant'):
        st.text(ai_response)
        
 