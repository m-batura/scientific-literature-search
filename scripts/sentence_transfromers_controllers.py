from sentence_transformers import SentenceTransformer

model = SentenceTransformer('ibm-granite/granite-embedding-107m-multilingual')

query_embedding = model.encode('Small test text')

print(query_embedding[:10])