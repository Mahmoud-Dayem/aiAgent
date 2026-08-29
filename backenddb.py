from typing import Literal , TypedDict ,Annotated
from langgraph.graph import StateGraph , START, END
from langchain_ollama import ChatOllama
from pydantic import Field, BaseModel
from langgraph.graph.message import BaseMessage , add_messages
from langchain_core.messages import HumanMessage ,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool 
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_ollama import  OllamaEmbeddings
from langchain_core.messages import SystemMessage
# from llmsource import llm
system_message = SystemMessage(
    content="""
You are Dayem AI, a helpful technical assistant.

You have access to the following tools:

1. search_pdf
   Searches the currently uploaded document using the RAG
   knowledge base.

2. calculator
   Performs mathematical calculations.

DOCUMENT RULES:

- When the user's question is related to the uploaded document,
  ALWAYS use the search_pdf tool before answering.

- Do not answer document-specific questions from your own memory.

- Use the information retrieved by search_pdf as the primary
  source for your answer.

- Preserve important technical details such as:
  alarm codes, parameter numbers, equipment names, settings,
  values, procedures, warnings, and troubleshooting steps.

- If the user asks about a specific code, alarm, parameter,
  model number, tag, or technical term, include that exact
  identifier in the search query.

- If the retrieved information does not contain enough
  information to answer the question, clearly state that
  the answer was not found in the uploaded document.

- Do not invent missing technical information.

CALCULATION RULES:

- Use the calculator tool when mathematical calculations
  are required.

GENERAL QUESTIONS:

- For normal conversation or questions unrelated to the
  uploaded document, answer normally without using search_pdf.
"""
)


import sqlite3
model = "qwen3.5:9b"
# Define tool
import numexpr
######################### Define Rag tool
# load pdf
from langchain_community.document_loaders import PyMuPDFLoader

### Create the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
# Global retriever
retriever = None

def load_document(file_path: str):

    global retriever

    print("\n==============================")
    print("📄 LOADING DOCUMENT")
    print("FILE:", file_path)
    print("==============================")

    # Load PDF
    loader = PyMuPDFLoader(file_path)

    documents = loader.load()

    print("Documents loaded:", len(documents))

    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Clean Chroma metadata
    chunks = filter_complex_metadata(chunks)

    print("Chunks created:", len(chunks))

     # -----------------------------------------
    # Create Chroma collection
    # -----------------------------------------
    vector_store = Chroma(
        collection_name="uploaded_document",
        embedding_function=embeddings
    )
    
    # -----------------------------------------
    # Add documents in batches
    # -----------------------------------------
    batch_size = 20

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        print(
            f"Embedding chunks "
            f"{i + 1} - {min(i + batch_size, len(chunks))}"
            f" / {len(chunks)}"
        )

        vector_store.add_documents(batch)

    # -----------------------------------------
    # Retriever
    # -----------------------------------------
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5
        })
    
    

     

    print("✅ Document ready for RAG")

    return len(documents), len(chunks)



# file_path = "intecont.pdf"

# loader = PyMuPDFLoader(file_path)

# documents = loader.load()



### Split the PDF into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = text_splitter.split_documents(documents)


# cleaned_chunks = filter_complex_metadata(chunks)
# # Create the FAISS vector database
# # vector_store = FAISS.from_documents(
# #     documents=chunks,
# #     embedding=embeddings
# # )
# vector_store = Chroma.from_documents(
#     documents=cleaned_chunks,
#     embedding=embeddings,
#     persist_directory="./instrumentation_chroma_db",
#     collection_name="instrumentation"
# )

# # Convert FAISS into a retriever 
# retriever = vector_store.as_retriever(
#     search_kwargs={
#         "k": 3
#     }
# )
# Build RAG TOOL
@tool
def search_pdf(query: str) -> str:
    """
    Search the uploaded PDF document for information
    relevant to the user's question.
    """

    global retriever

    if retriever is None:
        return (
            "No document has been uploaded yet. "
            "Please upload a PDF document first."
        )

    print("\n==============================")
    print("✅ SEARCH_PDF TOOL EXECUTED")
    print("QUERY:", query)
    print("==============================")

    docs = retriever.invoke(query)

    results = []

    for i, doc in enumerate(docs, start=1):

        page = doc.metadata.get("page")

        page_number = (
            page + 1
            if page is not None
            else "Unknown"
        )

        print(f"\n--- RESULT {i} ---")
        print("Page:", page_number)
        print(doc.page_content)

        results.append(
            f"Page {page_number}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(results)

#################### END of Rag Tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    
#------------- End calculator tool

tools = [calculator , search_pdf]
tools_node = ToolNode(tools)

#------ 

llm_no_tools = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    reasoning=False,
    num_ctx=8192,
    num_predict=512,
    keep_alive="30m")


llm = llm_no_tools.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState)->ChatState:
    
    messages = [ system_message, *state["messages"] ]
   
    response = llm.invoke(messages)
    
    return{
        "messages":[response]
    }
    
connection = sqlite3.connect(database="chatbot.db",check_same_thread=False)    
checkpoint = SqliteSaver(connection)
graph = StateGraph(ChatState)
# define nodes
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tools_node)
graph.add_conditional_edges('chat_node',tools_condition)

# define edges
graph.add_edge(START,'chat_node')
graph.add_edge('tools','chat_node')
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