from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

# Define the first node which is going to be the agent reasoning node.
# So it's going to receive the state which is a message state which is a dictionary that has the key of messages.
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """

    # And this is going to send the human message.
    # It's going to send it back to the LM. And then the LLM decides whether we need to use a function call and execute a
    # function, or it decides that it can output the answer.
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])

    return {"messages": [response]}

# And we also need to define the node with the relevant tools.
# So this is going to help Langgraph execute the tools that we need.
tool_node = ToolNode(tools)