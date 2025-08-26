import faiss
import numpy as np
import json
from google import genai
from google.genai import types

#constants.py - file with api_keys
import constants

papers_path = '../data/faiss/papers.faiss'
experimental_path = '../data/faiss/experimental.faiss'

def get_embedding(text_to_embed, client):
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text_to_embed,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    print('Recieved embedding')
    return result.embeddings[0].values

def add_to_exper_faiss(embedding):
    index = faiss.read_index(experimental_path)
    index.add(embedding)
    faiss.write_index(index, papers_path)
    print("Embeddings added to faiss index")
    return index.ntotal - 1

def add_to_faiss(embedding):
    index = faiss.read_index(papers_path)
    index.add(embedding)
    faiss.write_index(index, papers_path)
    print("Embeddings added to faiss index")
    return index.ntotal - 1

if __name__ == "__main__":
    # gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    # embedding = get_embedding('The article examines the issue of drug clustering. Initially, k classes are arbitrarily formed and theresulting training sample is pre-processed, then the similarities between the objects of each classare evaluated based on the proximity function and the criterion for evaluating the contribution ofobjects to the formation of their own class. Usually, it is in percentage and is the degree of mutualsimilarity of objects of each class. In the next steps of the algorithm, first, one object is takenfrom the first class, and by adding it to all k classes, the contribution of this object to this classis measured. The object will be left in the class which has the most contribution. This processis repeated several times in a row for all objects of the class. The process is stopped when thelocation of objects does not change and the degree of similarity exceeds the required percentage.As a result, the required clusters are formed.', gai)
    # print(len(embedding))
    index = faiss.IndexFlatL2(768)
    faiss.write_index(index, experimental_path)