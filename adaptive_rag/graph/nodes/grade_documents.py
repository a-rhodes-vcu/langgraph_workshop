from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

# So we're going to define a function which will receive the state.


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
   # supply original question and the documents
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    # And in that state we're going to have already the fetched documents.
    # We're going to iterate through all the documents.
    # And our grader chain is going to decide for each document whether it's relevant or not.
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            # append document to list because it's relevant to the question
            filtered_docs.append(d)
        else:
            # And finally, if we have found any document that's not relevant, we're going to change the web searching
            # flag to be true so we can go and later on search for that query.
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    # update graph state
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
