import streamlit as st
import backend

from langchain_core.messages import HumanMessage, AIMessageChunk


config = {
    "configurable": {
        "thread_id": "user_1"
    }
}

st.title("Dayem AI")
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import numpy as np
import streamlit as st
import numpy as np
import pandas as pd
import time

import uuid

# ---- side bar --

st.sidebar.title("My conversation")
 
 

user_input = st.chat_input("Enter your message:")

if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = []


def generate_uuid():
    print('generating uuid')
    
    new_uuid = str(uuid.uuid4())
    print(new_uuid)
    st.session_state.chat_threads.append(new_uuid)
    
st.button("New Chat",on_click=generate_uuid)  
  
for thread in st.session_state.chat_threads:
    print(thread)
    
    st.sidebar.write(thread)    
if user_input:

    # Save user message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Show user message
    with st.chat_message("user"):
        st.write(user_input)


    init = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }


    # Stream assistant response
    with st.chat_message("assistant"):
        ai_response = st.write_stream(
            message_chunk.content for message_chunk, metadata in  backend.chatbot.stream(
                init,
                config = config,
                stream_mode = 'messages'
            )
        )
      

  


    # Save final assistant response
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )
    
