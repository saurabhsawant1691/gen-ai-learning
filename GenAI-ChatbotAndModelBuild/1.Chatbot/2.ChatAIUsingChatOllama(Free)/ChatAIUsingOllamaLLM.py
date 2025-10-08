from langchain_ollama import OllamaLLM

llm = OllamaLLM(
  model='llama3',
  max_tokens=10
)

prompt = "What is the capital of France?"

response = llm.invoke(prompt)
print(response)