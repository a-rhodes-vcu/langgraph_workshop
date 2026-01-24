from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

# This function is going to receive the state.
# And it's going to return a dictionary.
def retrieve(state: GraphState) -> Dict[str, Any]:
    """
       Retrieve documents

       Args:
           state (dict): The current graph state

       Returns:
           state (dict): New key added to state, documents, that contains retrieved documents
       """
    print("---RETRIEVE---")
    question = state["question"]

    # To do the semantic search and get us all the relevant documents
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
