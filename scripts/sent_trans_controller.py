from sentence_transformers import SentenceTransformer
import scrap_sinteza
import numpy as np
import time

def save_model(model_name):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save('st_models')

def load_model(model_path):
    return SentenceTransformer(model_path, trust_remote_code=True)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def get_embedding()
# save_model("Alibaba-NLP/gte-multilingual-base")

if __name__ == "__main__":
    print('program start')
    time_stamp = time.time()
    model = load_model('gte-multilingual-base')
    print(f'time to load {(time.time() - time_stamp)}')
    # promt = 'automating literature search using web scrapping embeddings and vector search'
    # promt_embedding = model.encode(promt)
    papers = scrap_sinteza.scrap_papers('https://portal.sinteza.singidunum.ac.rs/issue/showAll/2025')
    time_stamp = time.time()
    paper_embeddings = np.array([model.encode(paper[0]) for paper in papers])
    # paper_similarities = np.array([])
    print(f'avg embed time {(time.time() - time_stamp) / paper_embeddings.shape[0]}')
    time_stamp = time.time()
    similarities = np.array([cosine_distance(paper_embeddings[0], paper_embedding) for paper_embedding in paper_embeddings])
    print(f'avg distance time {(time.time() - time_stamp) / paper_embeddings.shape[0]}')
    print(similarities.dtype)