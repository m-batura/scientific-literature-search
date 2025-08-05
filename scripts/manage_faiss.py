import faiss
import numpy as np
import json
from google import genai
from google.genai import types

#constants.py - file with api_keys
import constants

index_path = '../data/faiss/papers.faiss'

def get_embedding(text_to_embed, client):
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text_to_embed,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    print('Recieved embedding')
    return result.embeddings[0].values

def add_to_faiss(embedding):
    index = faiss.read_index(index_path)
    index.add(embedding)
    faiss.write_index(index, index_path)
    print("Embeddings added to faiss index")
    return index.ntotal - 1

#if __name__ == "__main__":
    # json_path = '../../data/json/math.json'
    # print('Opening JSON')
    # json_file = open(json_path, encoding='utf-8')
    # json_data = json.load(json_file)
    #
    # entries = json_data['papers'][:]
    # abstracts = []
    # for entry in entries:
    #     abstracts.append(entry['abstract'])
    #
    # json_file.close()
    #
    # gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    #
    # # list comprehension
    # print('list comprehension')
    # embeddings = [get_embedding(abstract, gai) for abstract in abstracts]
    # embeddings_array = np.array(embeddings).astype('float32')
    # dimension = embeddings_array.shape[1]
    #
    # index = faiss.IndexFlatL2(dimension)
    # index.add(embeddings_array)
    # faiss.write_index(index, index_path)