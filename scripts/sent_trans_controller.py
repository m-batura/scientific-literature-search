from sentence_transformers import SentenceTransformer
import time

def save_model(model_name):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save('st_models')

def load_model(model_path):
    return SentenceTransformer(model_path, trust_remote_code=True)

# def get_embedding()
# save_model("Alibaba-NLP/gte-multilingual-base")

if __name__ == "__main__":
    start = time.time()
    print(time.time() - start)
    model = load_model('st_models')
    print(time.time() - start)
    texts  = [
        'Enhanced Monte Carlo Schedule Analysis: Evaluation of the Open-Source Pert-Based Simulation Tool',
        'Artificial Intelligence in the Creative Industry: Strategic Implementation of Marketing Tools',
        'Microsoft Copilot as a Transformative Tool in Business: Opportunities and Challenges'
    ]
    embeddings = model.encode(texts)
    print(time.time() - start)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    print(time.time() - start)
