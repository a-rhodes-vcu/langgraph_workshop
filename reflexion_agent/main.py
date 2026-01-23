from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2

# Draft the response. So we want to call the first responder chain
# and we are going to give it when we invoke it
# in the messages key, the state in the messages place.
# So this is going to take all of the messages that we have in the graph and send it.
def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# And this is going to revise the answer based on the tool results, which was the Pydantic critique.
# And here, we simply want to take our reviser chain and we want to invoke it with all the messages we have so far.
def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# So the conditional edge is going to be called event loop
# and it's going to receive the state as an input,
# which is going to be a list of messages.

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:

    """Determine whether to continue or end based on iteration count."""
    #  Count the number of tool calls.
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits

    # if num_iterations is going to be greater than max iterations, which is two,
    # then we want to finish
    # So this is going to ensure us we are running this architecture only twice.
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

# StateGraph to receive the message state - list of messages.
builder = StateGraph(MessagesState)
# put all then nodes and edges together
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

print(graph.get_graph().draw_mermaid())



res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)
# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)
