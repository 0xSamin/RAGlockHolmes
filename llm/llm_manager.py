from langchain_ollama import ChatOllama

# Initialize the LLM
llm = ChatOllama(
    model="hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:UD-Q4_K_XL",
    temperature=0 # Keep it 0 for RAG to minimize hallucination
)

# Test the connection
response = llm.invoke("Hello! Are you ready to analyze some papers?")
print(response.content)
