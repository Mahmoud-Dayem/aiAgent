import streamlit as st
import backend
import uuid

from langchain_core.messages import HumanMessage


st.title("Dayem AI")

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

# List of all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []


# Dictionary:
# {
#     thread_uuid_1: [messages...],
#     thread_uuid_2: [messages...]
# }
if "thread_histories" not in st.session_state:
    st.session_state.thread_histories = {}


# Currently selected conversation
if "current_thread" not in st.session_state:
    st.session_state.current_thread = None


# =========================================================
# CREATE NEW CHAT
# =========================================================

def create_new_chat():

    new_uuid = str(uuid.uuid4())

    # Add new thread
    st.session_state.chat_threads.append(new_uuid)

    # Create empty message history for this thread
    st.session_state.thread_histories[new_uuid] = []

    # Select the new thread
    st.session_state.current_thread = new_uuid


# =========================================================
# CREATE FIRST CHAT AUTOMATICALLY
# =========================================================

if st.session_state.current_thread is None:
    create_new_chat()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("My Conversations")


# New Chat button
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()


st.sidebar.divider()


# Display all previous threads
for thread in st.session_state.chat_threads:

    # Short version of UUID for display
    display_name = thread[:8]

    if st.sidebar.button(
        f"💬 {display_name}",
        key=f"thread_{thread}"
    ):
        st.session_state.current_thread = thread
        st.rerun()


# =========================================================
# CURRENT THREAD
# =========================================================

current_thread = st.session_state.current_thread


# LangGraph configuration
config = {
    "configurable": {
        "thread_id": current_thread
    }
}


# Get history belonging to current thread
message_history = st.session_state.thread_histories[current_thread]


# =========================================================
# DISPLAY CURRENT CHAT HISTORY
# =========================================================

for message in message_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input("Enter your message:")


if user_input:

    # -----------------------------------------------------
    # Save user message
    # -----------------------------------------------------

    message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    with st.chat_message("user"):
        st.write(user_input)


    # -----------------------------------------------------
    # Send message to LangGraph
    # -----------------------------------------------------

    init = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }


    # -----------------------------------------------------
    # Stream assistant response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        ai_response = st.write_stream(

            message_chunk.content

            for message_chunk, metadata
            in backend.chatbot.stream(
                init,
                config=config,
                stream_mode="messages"
            )

            if message_chunk.content
        )


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    message_history.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )