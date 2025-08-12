import numpy as np
from google import genai

import constants
import scrap_papers as sp
import manage_faiss as faiss

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def lang_similarity_exp(path, id, client):
    paper_en = sp.get_kaznu_paper(path, id, 'en_US')
    abstract_en = paper_en[1]
    embedding_en = faiss.get_embedding(abstract_en, client)
    paper_ru = sp.get_kaznu_paper(path, id, 'ru_RU')
    abstract_ru = paper_ru[1]
    embedding_ru = faiss.get_embedding(abstract_ru, client)
    paper_kz = sp.get_kaznu_paper(path, id, 'uk_UA')
    abstract_kz = paper_kz[1]
    embedding_kz = faiss.get_embedding(abstract_kz, client)

    dist_en_ru = cosine_distance(embedding_en, embedding_ru)
    dist_en_kz = cosine_distance(embedding_en, embedding_kz)
    dist_ru_kz = cosine_distance(embedding_ru, embedding_kz)

    print(f"en-ru: {dist_en_ru:.4f}, en-kz: {dist_en_kz:.4f}, ru-kz: {dist_ru_kz:.4f}")

def compare_paper_to_citations(paper_abstract, citatons_abstracts, client):
    paper_embedding = faiss.get_embedding(paper_abstract, client)
    for citation_abstract in citatons_abstracts:
        citation_embedding = faiss.get_embedding(citation_abstract, client)
        print(cosine_distance(paper_embedding, citation_embedding))
    print('a')

if __name__ == "__main__":
    gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    # lang_similarity_exp('https://bm.kaznu.kz/index.php/kaznu/', 1608, gai)
    # lang_similarity_exp('https://philart.kaznu.kz/index.php/1-FIL/', 4918, gai)
    # lang_similarity_exp('https://bulletin-psysoc.kaznu.kz/index.php/1-psy/', 2106, gai)

