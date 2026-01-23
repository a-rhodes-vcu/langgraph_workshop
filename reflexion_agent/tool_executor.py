from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

# get five results back
tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    # Run the query with the batch function which is going to run them concurrently.
    return tavily_tool.batch([{"query": query} for query in search_queries])

# Now I remind you this tool is going to examine the state.
# It's going to check the last message.
# And if there is a tool call it's going to execute the relevant tool call.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
