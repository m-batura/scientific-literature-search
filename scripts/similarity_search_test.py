from google import genai
import faiss
import json
import numpy as np
from google.genai import types

from manage_faiss import get_embedding
import constants
import manage_db as db

# json_path = '../../data/json/math.json'
# print('Opening JSON')
# json_file = open(json_path, encoding='utf-8')
# json_data = json.load(json_file)
#
# entries = json_data['papers'][:]

gai = genai.Client(api_key=constants.GEMINI_API_KEY)
input_string = 'негативных эмоциональных поведенческих реакций\n'
# input_string = 'Write a related work section based on scientific paper abstracts. \n'
print('get promt embedding')
query_embedding = get_embedding(input_string, gai)
query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

k = 20
index = faiss.read_index('../../data/faiss/papers.faiss')
print('vector search')
distances, indices = index.search(query_embedding, k)

for i in range(len(indices[0])):
    print(indices[0][i])
    abstract = db.get_paper_by_embedding_id(int(indices[0][i]))
    print(f'{indices[0][i]}-{distances[0][i]:.3f}-{abstract}')

# prompt = 'Напиши литературный обзор на основе абстрактов на тему: \'' + input_string + '\'\nНе все абстаркты относятся к теме, исключи неподходящии. Добавь номера цитируемых абстрактов в конец абзацев\n\n'
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