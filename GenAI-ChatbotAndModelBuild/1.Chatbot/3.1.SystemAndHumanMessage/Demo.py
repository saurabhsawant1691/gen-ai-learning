from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
  model='llama3',
  max_tokens=100
)

# First conversation - Programming context
context = SystemMessage(
    content="You are a Java Developer."
)
human_message = HumanMessage("Tell me a joke.")
response = llm.invoke([context, human_message])
print(response.content)

# ai_message = AIMessage(response.content)

print("-----------------------------------")

context = SystemMessage(
  content="You are a Python developer."
)
human_massage=HumanMessage("Tell me a joke.")
response = llm.invoke([context, human_massage])
print(response.content)
