# ReAct (Reasoning and Acting) agent using LangGraph and LangChain
- This example demonstrates how to create a ReAct agent using LangGraph and LangChain.
- Required install
```
pip install langgraph==0.2.74
pip install langchain-community langchain-core
```

![Screenshot 2568-05-01 at 21 59 19](https://github.com/user-attachments/assets/3e853f76-44f5-4767-ab2a-61dc6763b371)

The result is
```
The square root of 101 is approximately 10.05.
```

If we use only `print(messages)`
The result it will be
```
{'messages':
  [HumanMessage(content='What is the square root of 101?'.....
  AIMessage(content=''...
  ToolMessage(content='Answer: 10.04987562112089'...
  AIMessage(content='The square root of 101 is approximately 10.05.'
}
```
