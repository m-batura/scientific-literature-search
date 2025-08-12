from google import genai
import faiss
import json
import numpy as np
from google.genai import types

from manage_faiss import get_embedding
import constants
import manage_db as db

def literature_reivew(model, topic):
    # topic = 'Write a related work section based on scientific paper abstracts. \n'
    print('get promt embedding')
    query_embedding = get_embedding(topic, model)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    k = 20
    index = faiss.read_index('../data/faiss/papers.faiss')
    print('vector search')
    distances, indices = index.search(query_embedding, k)

    for i in range(len(indices[0])):
        try:
            abstract = db.get_paper_by_embedding_id(int(indices[0][i]))
            print(f'{indices[0][i]}-{distances[0][i]:.3f}-{abstract}')
        except TypeError:
            print('Entry not found')

    # prompt = 'Напиши литературный обзор на основе абстрактов на тему: \'' + topic + '\'\nНе все абстаркты относятся к теме, исключи неподходящии. Добавь номера цитируемых абстрактов в конец абзацев\n\n'
    # for i in range(len(indices[0])):
    #     print(f'{i} - {entries[indices[0][i]]['abstract']}')
    #     prompt = prompt + f'{i} - {entries[indices[0][i]]['abstract']}\n'
    #
    # client = genai.Client(api_key=constants.GEMINI_API_KEY)
    #
    # print('get llm answer')
    # response = client.models.generate_content(
    #     model="gemini-2.0-flash-lite",
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         temperature=0,
    #         system_instruction="Ты пишешь литературный обзор для научной статьи. Дай простой ответ без вступлений и выводов"
    #     )
    # )
    # print(response.text)

if __name__ == "__main__":
    gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    topic = 'влияние искусственного интеллекта на развитие языка\n'
    literature_reivew(gai, topic)