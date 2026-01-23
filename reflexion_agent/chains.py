import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
# we want to import from linked chain output parsers, some output parsers that would handle the output
# from the function calling from OpenAI.
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

# We want to import the chat prompt template.
# this is going to hold all of our history of our agent iterations. So we're going to append messages through that.
# the message placeholder is useful to being a placeholder for new messages.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="o4-mini")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


# This prompt template is also going to be used by our Reviser node, the node which is going to take
# all of the information and it's going to rewrite the article.

# The first part is a placeholder of first instructions.
# And here we're going to plug in to simply write a 250 word essay.

#The second is critique is going to be used later by the reviser agent.

# The third part is recommended search queries to research information and improve your answer.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
# When we invoke this prompt template.
# Then we want to plug in here the current date.
# So we're simply going to use a lambda function that will output us today's date.
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# First responder chain which is going to take our prompt template.
# And it's going to pipe it into the LLM GPT four turbo.
#Uses the answer question object as a tool for the function calling
# This will force the LLM to always use the answer question tool, thus grounding the response to the object that we want to receive.
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# Will inherit from the answer creation class.
# So it will have answer reflection search queries.
# It will also have the references field which is going to be a list of strings.
# And those strings are going to be citations of URLs mostly that we'll get from the search engine.
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    chain = (
    # Send the human message in the message's key.
    # So that would plug the messages placeholder that we initialized in our prompt.
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)


