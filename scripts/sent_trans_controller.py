from sentence_transformers import SentenceTransformer
import numpy as np

path_to_gte = '.\\st_models\\gte-multilingual-base'

def save_model(model_name='Alibaba-NLP/gte-multilingual-base'):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save(path_to_gte)

def load_model(model_path=path_to_gte):
    return SentenceTransformer(model_path, trust_remote_code=True)

def cosine_distance(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

if __name__ == "__main__":
    save_model()
    0