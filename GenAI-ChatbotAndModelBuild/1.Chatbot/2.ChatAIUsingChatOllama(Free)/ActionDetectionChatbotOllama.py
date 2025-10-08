from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model='llama3',
    max_tokens=100
)

def detect_action(prompt):
    system_msg = SystemMessage(
        content="""You are an action detector. 
        Analyze the user input and respond with exactly one word from these options:
        - IMAGE (for image creation/generation requests)
        - AUDIO (for audio/speech related requests)
        - TRANSLATION (for language translation requests)
        - TEXT (for general queries)"""
    )
    
    human_msg = HumanMessage(
        content=f"Classify this request: {prompt}"
    )
    
    response = llm.invoke([system_msg, human_msg])
    return response.content

while True:
    user_input = input("\nEnter your prompt: ")
    if user_input.lower() == 'exit':
        break
    
    action = detect_action(user_input)
    print(f"Detected Action: {action}")

print("\nGoodbye!")