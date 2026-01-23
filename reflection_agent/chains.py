# the chat prompt template is going to hold our content that we either sent to the LLM
# as humans, or that we receive back from the LLM as an answer that is tagged as an AI.
# And the second class is the messages placeholder, which is going to give us flexibility to put here
# a placeholder for future messages that we're going to get.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Then this prompt is supposed to act as our critique.
# So it's going to review the output.
# And in this case is a Twitter post and it's going to criticize it.
# So it's going to say how it can be better and a suggestion to improve it.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# Now we're going to be writing the generation prompt.
# And in our agent architecture, the generation prompt is going to generate the tweets that are going
# to be revised over and over again after the feedback we get from the reflection prompt.
# So it's going to revise the tweet until it gets the perfect tweet.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
