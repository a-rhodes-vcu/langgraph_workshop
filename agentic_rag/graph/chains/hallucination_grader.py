from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)




class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    # if answer is grounded in facts then return boolean yes, else no

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# cast the LLM answer into a boolean.
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
# create the hallucination greater chain, which is going to take the hallucination prompt.
# And it's going to pipe it to the structured LLM grader.
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
