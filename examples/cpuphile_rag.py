import langchain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceInference
from langchain.retrievers import VectorStoreRetriever
from langchain.vectorstores import FAISS

# Load the Wikipedia article.
loader = TextLoader("olympia-stadium-berlin.txt")
documents = loader.load()

# Create embeddings.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store.
vectorstore = FAISS.from_documents(documents, embeddings)

# Create a retriever.
retriever = VectorStoreRetriever(vectorstore=vectorstore)

# Create an LLM.
llm = HuggingFaceInference(model_name="facebook/opt-350m")

# Create a chain.
chain = langchain.RetrievalQA(retriever=retriever, llm=llm)

# Ask a question.
query = "What is the capacity of the Olympia Stadium in Berlin?"
print(chain.run(query))

