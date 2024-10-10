from sentence_transformers import SentenceTransformer
sentences = ["hi there"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

# Check the shape of the embeddings
print("dimensions: ")
print(embeddings.shape)
