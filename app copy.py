import streamlit as st
import backend

from langchain_core.messages import HumanMessage, AIMessageChunk


config = {
    "configurable": {
        "thread_id": "user_1"
    }
}

st.title("Dayem AI")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Show previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Enter your message:")


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

        placeholder = st.empty()

        ai_response = ""

        for message, metadata in backend.chatbot.stream(
            init,
            config,
            stream_mode="messages"
        ):

            if isinstance(message, AIMessageChunk):

                if message.content:

                    ai_response += message.content

                    placeholder.markdown(
                        ai_response + "▌"
                    )

        placeholder.markdown(ai_response)


    # Save final assistant response
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )