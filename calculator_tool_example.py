import ast
import operator
from typing import List, Any, Dict, Optional
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.agents import tool_calling
import sys

# -----------------------------------------------------------------------------
# 1. Define the Calculator Tool Logic
# -----------------------------------------------------------------------------

def safe_evaluate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Only allows numbers, +, -, *, /, parentheses.
    """
    if not expression or not isinstance(expression, str):
        raise ValueError("Expression must be a non-empty string.")
    
    # Compile to AST to ensure safety (restrict to literals and operators)
    parsed = ast.parse(expression, mode='eval')
    
    # Filter nodes to ensure only math literals exist
    safe_ops = {ast.Num: lambda n: n.n if hasattr(n, 'n') else n.d, 
                ast.Constant: lambda c: c.value}
    
    def extract_value(node):
        for child in ast.walk(node):
            if isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.UAdd) or isinstance(child.op, ast.USub):
                 continue
            # We just allow numbers and basic operators. 
            # For simplicity in this demo, we will use a controlled Eval 
            # but restrict allowed character set first.
        return parsed
    
    # Simplest robust approach for demo: 
    # 1. Verify chars are restricted to digits and math symbols
    if not all(c in "0123456789+-*/(). " for c in expression):
         raise ValueError("Invalid characters in expression.")

    try:
        result = eval(expression)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero."
    except SyntaxError:
        return "Error: Invalid syntax. Please check your numbers and symbols."
    except Exception as e:
        return f"Error: Could not compute expression ({str(e)})."


def run_calculator(expression: str) -> str:
    """Wrapper for the tool."""
    return safe_evaluate(expression)


# Create the Tool Definition
calculator_tool = BaseTool(
    name="calculator",
    func=run_callector, # Correcting typo in thought process: func=run_calculator
    description="Use this to perform simple mathematical operations (addition, subtraction, multiplication, division). "
                "Provide an expression string like '2 + 5' or '100 * 4'.",
    schema={
        "name": "calculator",
        "description": "Calculates the result of a math operation.",
        "parameters": {
            "properties": {
                "expression": {
                    "type": "string"
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
            "type": "object"
        }
    }
)

# Fixing the function reference above
calculator_tool.func = run_calculator 

# -----------------------------------------------------------------------------
# 2. Define the LangGraph State and Nodes
# -----------------------------------------------------------------------------

class AgentState(StateGraph.State):
    """State dictionary for the graph."""
    messages: list[Dict[str, Any]]
    response: Optional[str] = None

def model_node(state: AgentState) -> str:
    # Initialize the ChatModel with the calculator tool.
    # Note: For a real app, you'd use ChatOpenAI and pass tools.
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # In LangGraph, we usually wrap this in an Agent logic handler 
    # or simply call the LLM directly if it's a simple router agent.
    # Here we construct a simple prompt that uses tools.
    
    from langchain_core.prompts import ChatPromptTemplate
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}")
    ])
    
    chain = template | llm.with_structured_output(lambda x: str) # Standard LLM usage
    
    # To use tools properly with LangGraph + LangChain LLM, 
    # we typically use the `with_structured_output` or `tool_calling`.
    # For this example, let's simulate the tool injection manually via 
    # a simplified Agent loop structure inside LangGraph.
    
    # However, to make it runnable in this script without complex agent setups:
    # We will use LangChain's AgentExecutor logic to manage tools inside the graph.
    
    # Note: This specific node is a placeholder for where the LLM thinks.
    # In practice, you would define an `Agent` node that looks like this:
    agent = llm.with_structured_output(lambda x: str)
    
    # For simplicity in this standalone example, we will return messages to show flow.
    return state["messages"] # Return current message to process next

def end_node(state: AgentState):
    return {"response": "Response ready."}

# -----------------------------------------------------------------------------
# 3. Build the Graph
# -----------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("agent", model_node)
workflow.add_node("end", end_node)

workflow.set_entry_point("agent")

# Define edges: Agent can pass to itself (looping for tool calls) or to end
workflow.add_edge("end", END)

# Add the calculator tool to the agent's context
# In a real LangGraph implementation using `langchain.agents`, you would pass tools
# directly to the AgentExecutor. Here, we demonstrate the Tool definition structure.

graph = workflow.compile(checkpointer=MemorySaver())

# -----------------------------------------------------------------------------
# 4. Execute and Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Calculator LangGraph Demo ---\n")
    
    # This is a raw execution example because connecting an actual LLM
    # requires API keys. We will mock the LLM response for demonstration, 
    # or explain where to plug in the real `ChatOpenAI`.
    
    # To run with real LLM:
    #   from langchain.agents import create_tool_calling_agent
    #   agent = create_tool_calling_agent(llm, [calculator_tool], prompts)
    #   executor = AgentExecutor(agent=agent, tools=[calculator_tool])
    #   
    # Since we are running this standalone:
    print("1. Creating Calculator Tool Instance...")
    tool_instance = calculator_tool
    
    # Test the logic directly first
    test_expr = "5 + 10"
    print(f"   Testing function directly with '{test_expr}' -> {tool_instance.run(test_expr)}")
    
    # To actually run the Graph, you need to inject the tools into the LLM 
    # via the ChatModel configuration (e.g., chat_model.bind_tools([calculator_tool])).
    
    print("\n2. Setup Complete.")
    print("   The tool 'calculator' is now ready to be registered with an AgentNode.")
