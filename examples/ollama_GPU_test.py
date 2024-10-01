from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1")

response = llm.invoke("tell me a little story while I monitor top and nvtop and view my GPU usage when using this model")
print(response)
