import streamlit as st
import backenddb
import uuid
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os


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
# =========================================================
# DOCUMENT UPLOAD
# =========================================================

st.sidebar.subheader("Knowledge Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Prevent rebuilding embeddings on every Streamlit rerun
    if (
        "loaded_document" not in st.session_state
        or st.session_state.loaded_document != uploaded_file.name
    ):

        with st.sidebar.status(
            "Processing document...",
            expanded=True
        ) as status:

            # Save uploaded file locally
            os.makedirs(
                "uploaded_documents",
                exist_ok=True
            )

            file_path = os.path.join(
                "uploaded_documents",
                uploaded_file.name
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("📄 Loading PDF...")

            pages, chunks = backenddb.load_document(
                file_path
            )

            st.write(
                f"Pages: {pages}"
            )

            st.write(
                f"Chunks: {chunks}"
            )

            st.session_state.loaded_document = (
                uploaded_file.name
            )

            status.update(
                label="Document ready",
                state="complete"
            )

    st.sidebar.success(
        f"📚 {uploaded_file.name}"
    )

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

    # with st.chat_message("assistant"):
    #     ai_response = st.write_stream(
    #         message_chunk.content
    #         for message_chunk, metadata
    #         in backenddb.chatbot.stream(
    #             init,
    #             config=config,
    #             stream_mode="messages"
    #         )
    #         if message_chunk.content
    #     )
    
 
    with st.chat_message("assistant"):

        def stream_response():

            for message_chunk, metadata in backenddb.chatbot.stream(
                init,
                config=config,
                stream_mode="messages"
            ):

                # Debugging
                print("\n==============================")
                print("MESSAGE TYPE:")
                print(type(message_chunk))

                print("\nCONTENT:")
                print(message_chunk.content)

                print("\nTOOL CALLS:")
                print(getattr(message_chunk, "tool_calls", None))

                print("\nTOOL CALL CHUNKS:")
                print(getattr(message_chunk, "tool_call_chunks", None))

                print("\nMETADATA:")
                print(metadata)

                # Display tool call
                if getattr(message_chunk, "tool_calls", None):

                    for tool_call in message_chunk.tool_calls:

                        st.info(
                            f"🔧 Tool used: {tool_call['name']}"
                        )

                # Display streamed answer
                if message_chunk.content:

                    yield message_chunk.content


        ai_response = st.write_stream(
            stream_response()
        )
 


    # Refresh so this new conversation appears in sidebar.
    st.rerun()