from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT= "act"
# this is for future when we're going to reference the last message between the user and the agent.
LAST = -1


# So if the last message is going to be a tool call then we want to go to the node.
# And if not we want to go to end.

# So this means that the LLM, the reasoning engine, decided that we need to invoke a tool and it has
# all the information of the arguments.
# Then we want to go and execute the tool node which will execute that tool.
# And if there isn't a tool call, then we simply want to end everything.
# this is a heuristic that the LM was able to answer the question with or without a tool invocation.
def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# give it the message graph.
flow = StateGraph(MessagesState)

# we want to add the node of agent reason and the node of act.
flow.add_node(AGENT_REASON, run_agent_reasoning)
# set entry point.
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

# add the conditional edge from the agent reason node to the node and to the unknown depending on
# a certain functionality.
# And the third argument is going to be a dictionary of mapping between the string end to end from the
# string act to act.
# And this mapping really tells Langgraph that whatever comes from the 'should continue function',
# which is going to be some kind of string. And we're going to return from that function the answering or the x ring.
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

# define the edge from act to agent reason.
# Because after we invoke the tool we want the agent to reason and to figure out whether it needs to return
# the answer or to run another tool call.
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)