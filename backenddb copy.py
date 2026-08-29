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

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_ollama import  OllamaEmbeddings
from langchain_core.messages import SystemMessage
# from llmsource import llm
system_message = SystemMessage( content="""
                               You are Dayem AI, a technical assistant. 
                               You have access to these tools: 1. search_pdf
                               Use this tool to search the INTECONT PDF
                               knowledge base. 2. calculator Use this 
                               tool for mathematical calculations. 
                               IMPORTANT RULES: - If the user's question 
                               may be related to INTECONT, weighing systems,
                               belt weighers, feeders, controllers, calibration, 
                               commissioning, parameters, maintenance, troubleshooting, 
                               alarms, or information contained in the loaded PDF, ALWAYS 
                               call search_pdf before answering. - Do not 
                               answer PDF-related questions from your own
                               memory. - After search_pdf returns information,
                               answer the user using the retrieved PDF content. 
                               - If the PDF search does not contain enough
                               information, clearly say that the information was not 
                               found in the PDF. - Use calculator when mathematical
                               calculation is required. """ )


import sqlite3
model = "qwen3.5:9b"
# Define tool
import numexpr
######################### Define Rag tool
# load pdf
from langchain_community.document_loaders import PyMuPDFLoader

file_path = "intecont.pdf"

loader = PyMuPDFLoader(file_path)

documents = loader.load()

print("Documents loaded:", len(documents))

### Split the PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

### Create the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Create the FAISS vector database
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Convert FAISS into a retriever 
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)
# Build RAG TOOL
@tool
def search_pdf(query: str) -> str:
    """
    Search the PDF knowledge base for information relevant
    to the user's question.
    """
    print("\n==============================")
    print("✅ SEARCH_PDF TOOL EXECUTED")
    print("QUERY:", query)
    print("==============================\n")

    docs = retriever.invoke(query)

    results = []

    for doc in docs:
        page = doc.metadata.get("page", "Unknown")

        results.append(
            f"Page {page + 1}\n"
            f"{doc.page_content}"
        )
        print(doc.page_content)


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