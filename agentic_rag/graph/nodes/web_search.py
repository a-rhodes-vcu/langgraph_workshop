from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
# get three web searches returned
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """receive the state and return a dictionary"""
    print("---WEB SEARCH---")
    # extract the question and documents from the graph state.
    question = state["question"]
    # we have already filtered out non-relevant documents, everything in the documents list are going to be relevant for our query.
    if "documents" in state: # if the route to web search in first time then give error
        documents = state["documents"]
    else:
        documents = None
        
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    # get one huge string from search results
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    # And what we want to do is to take all the content from all the elements of this list, and to combine
    # them into one document of length chain.
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        # if relevant documents then append to documents list
        documents.append(web_results)

    else:
        # if no relevant documents found then just return the web results as one list
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
