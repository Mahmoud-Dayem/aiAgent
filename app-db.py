import streamlit as st
import backenddb
import uuid

from langchain_core.messages import HumanMessage


st.title("Dayem AI")


# Current selected conversation only
if "current_thread" not in st.session_state:
    st.session_state.current_thread = None


def create_new_chat():
    # It becomes a SQLite thread after the first message is sent.
    st.session_state.current_thread = str(uuid.uuid4())


# Load saved thread IDs from SQLite
saved_threads = backenddb.get_all_threads()


# Create first empty chat
if st.session_state.current_thread is None:
    create_new_chat()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

st.sidebar.divider()

for thread_id in saved_threads:
    display_name = thread_id[:8]

    if st.sidebar.button(
        f"💬 {display_name}",
        key=f"thread_{thread_id}"
    ):
        st.session_state.current_thread = thread_id
        st.rerun()


# =========================================================
# CURRENT THREAD
# =========================================================

current_thread = st.session_state.current_thread

config = {
    "configurable": {
        "thread_id": current_thread
    }
}

# Load messages from SQLite
message_history = backenddb.get_thread_messages(current_thread)


# =========================================================
# DISPLAY SAVED HISTORY
# =========================================================

for message in message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input("Enter your message:")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    init = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }

    with st.chat_message("assistant"):
        ai_response = st.write_stream(
            message_chunk.content
            for message_chunk, metadata
            in backenddb.chatbot.stream(
                init,
                config=config,
                stream_mode="messages"
            )
            if message_chunk.content
        )

    # Refresh so this new conversation appears in sidebar.
    st.rerun()