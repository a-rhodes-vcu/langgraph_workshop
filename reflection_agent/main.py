# Type dict is a type dictionary which creates a structured dictionary with hints for the keys, needed for the state schema.
# lang graph would require typed state definitions to know which data flows through
# and out of the graph.
# Annotated is going to help add metadata to those type hints.
from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

# abstract base class for all message types in link chain,
# will be a type hint for the messages list.
# And this is going to ensure safety for different message types, whether it's a human message, an AI
# message, or a system message.

# human message to represent a message from a user.
# distinguish between the user content from the AI responses.
from langchain_core.messages import BaseMessage, HumanMessage
# the end which is a special constant to mark the graph termination.
# So this is the ending node of the graph.


# And the state is simply going to be a data structure, usually dictionary or a pedantic class, which
# is going to be maintained through the entire execution, and it's going to hold the information of the
# execution.
# We can store their intermediate results, LLM responses, and basically everything we can think of,
# we can store there.
# It's very flexible, and every node that is going to run is going to have access to this state.
from langgraph.graph import END, StateGraph

# Now this is a link graph reducer function.
# And the entire goal of this function is to ensure new messages are appended to the existing conversation
# history instead of replacing it.
from langgraph.graph.message import add_messages

# chains which are going to be nodes in our graph.
from chains import generate_chain, reflect_chain

# So this is the goal of this state here.
# Simply a data structure to hold all of those list of messages here.
class MessageGraph(TypedDict):
    # Here is metadata that will tell
    # Langgraph how to handle state updates.
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

# And the first node is going to be called the generation node.
# The input for this node is going to be the state which is type of message graph.
# And the first message is going to be the user input here.
# And after it every message is going to be an AI message.

# And this node is simply going to run the generation chain, which we remember from the previous video,
# and it's going to invoke it with all the messages that we have so far.
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

# And once we label this text here, this critique, when we label it as a human message to get a better result
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

# So this is simply to tell a graph what's going to be the state how to update it.
builder = StateGraph(state_schema=MessageGraph)
# So we'll start and create the generation node.
# And the first argument is going to be the name which is going to be the string generate.
# And the node a logic is going to be the generation node function.
builder.add_node(GENERATE, generation_node)
# So we have one deterministic edge from the reflect node to the generate node.
builder.add_node(REFLECT, reflection_node)
# And after we reflect, we want to always go to the generate node to generate a new tweet, which is
# based on the reflection we got from the reflect node.

# And now we want to tell the graph that the first node that we want to execute is to be the generation node.
# And we do this by using the method set entry point and giving it a node name.
# And that node name is going to be generate.
builder.set_entry_point(GENERATE)

# And this function is going to receive the state.
# And the output of this function is going to be a string which is going to be the node name.
# So this function is going to be called every time after we run the node.
# And the output is going to telegraph where to go next, either to go to the reflection node or to go
# to end everything.
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)
