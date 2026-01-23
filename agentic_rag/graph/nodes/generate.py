from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState



# And the generation node is going to be the last node that is going to be executed.
# We execute this node after we already retrieve the information, the relevant documents, after we filtered
# out the documents that were not relevant to our query, and even performed a search for the question
# that we want to answer.
# So after we have all the documents, we can augment the original query.
# And now it's time to generate.
def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # send question and documents and get response from LLM
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
