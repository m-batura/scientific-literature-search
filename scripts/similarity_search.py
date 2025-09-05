import numpy as np

from gai_controller import get_embedding
import faiss_controller
import constants
import sqlite_controller as db

def literature_reivew(topic):
    # topic = 'Write a related work section based on scientific paper abstracts. \n'
    print('get promt embedding')
    query_embedding = get_embedding(topic)

    # move to faiss_controller
    k = 10
    print('vector search')
    distances, indices = faiss_controller.similarity_search(faiss_controller.experimental_path, query_embedding, k)

    for i in range(len(indices[0])):
        try:
            print(indices[0][i], distances[0][i])
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
    topic = 'The article examines the issue of drug clustering. Initially, k classes are arbitrarily formed and the resulting training sample is pre-processed, then the similarities between the objects of each class are evaluated based on the proximity function and the criterion for evaluating the contribution of objects to the formation of their own class. Usually, it is in percentage and is the degree of mutual similarity of objects of each class. In the next steps of the algorithm, first, one object is taken from the first class, and by adding it to all k classes, the contribution of this object to this class is measured. The object will be left in the class which has the most contribution. This process is repeated several times in a row for all objects of the class. The process is stopped when the location of objects does not change and the degree of similarity exceeds the required percentage. As a result, the required clusters are formed.'
    literature_reivew(topic)