# This code demonstrates how to create a custom tool for a LangChain agent using the React framework.
# This example shows how to create a custom tool that generates a financial report for a company.
# The custom tool calculates the net income based on the provided revenue and expenses.
import os
from langchain_community.agent_toolkits.load_tools import load_tools                                                                        
from langchain_openai import ChatOpenAI        
import httpx  
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(verify=False)
)

@tool
def financial_report(company_name: str, revenue: int, expenses: int) -> str:
  """Generate a financial report for a company that calculates net income."""
  net_income = revenue - expenses

  report = f"Financial Report for {company_name}:\n"
  report += f"Revenue: ${revenue}\n"
  report += f"Expenses: ${expenses}\n"
  report += f"Net Income: ${net_income}\n"
  return report

agent_custom_tool = create_react_agent(llm, [financial_report])

messages = agent_custom_tool.invoke({"messages": [("human", f"""TechStack generated made $10 million
 with $8 million of costs. Generate a financial report.""")]})

print(messages['messages'][-1].content)