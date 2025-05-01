# ReAct (Reasoning and Acting) agent using LangGraph and LangChain
# This example demonstrates how to create a ReAct agent using LangGraph and LangChain.
# The ReAct agent is designed to perform reasoning and acting tasks, such as answering questions and performing calculations.
import os
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(verify=False)
)

# This example uses the `llm-math` tool to perform mathematical calculations.
tools = load_tools(["llm-math"], llm=llm)

# The ReAct agent is created using the `create_react_agent` function from the `langgraph.prebuilt` module.
agent = create_react_agent(llm, tools)

# The agent is then invoked with a message to perform a calculation.
messages = agent.invoke({"messages": [("human", "What is the square root of 101?")]})
print(messages['messages'][-1].content)