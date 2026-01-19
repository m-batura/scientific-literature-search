from sentence_transformers import SentenceTransformer

def save_model(model_name):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save('st_models')

def load_model(model_path):
    return SentenceTransformer(model_path, trust_remote_code=True)

# save_model("Alibaba-NLP/gte-multilingual-base")

model = load_model('st_models')
query_embedding = model.encode('Small test text')
print(query_embedding[:10])