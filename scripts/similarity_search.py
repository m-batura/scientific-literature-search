from gai_controller import get_embedding
import faiss_controller

def find_k_similar_texts(topic, k):
    print('get promt embedding')
    query_embedding = get_embedding(topic)

    print('vector search')
    distances, indices = faiss_controller.similarity_search(faiss_controller.experimental_path, query_embedding, k)

    for i in range(len(indices[0])):
        try:
            print(indices[0][i], distances[0][i])
        except TypeError:
            print('Entry not found')

if __name__ == "__main__":
    title = 'A New Hybrid Descent Algorithm for Large-Scale Nonconvex Optimization and Application to Some Image Restoration Problems'
    find_k_similar_texts(title, 10)
    topic = '''Conjugate gradient methods are widely used and attractive for large-scale unconstrained smooth optimization problems, with simple computation, low memory requirements, and interesting theoretical information on the features of curvature. Based on the strongly convergent property of the Dai–Yuan method and attractive numerical performance of the Hestenes–Stiefel method, a new hybrid descent conjugate gradient method is proposed in this paper. The proposed method satisfies the sufficient descent property independent of the accuracy of the line search strategies. Under the standard conditions, the trust region property and the global convergence are established, respectively. Numerical results of 61 problems with 9 large-scale dimensions and 46 ill-conditioned matrix problems reveal that the proposed method is more effective, robust, and reliable than the other methods. Additionally, the hybrid method also demonstrates reliable results for some image restoration problems.'''
    find_k_similar_texts(topic, 10)

# calling google ai to write review

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