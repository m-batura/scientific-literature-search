# from gai_controller import get_embedding
# import faiss_controller
import pandas as pd
import scrap_sinteza as sinteza
import sent_trans_controller as stc
import time

path_to_json = '.\\titles.json'
path_to_parquet = '.\\titles.parquet'

def save_titles_embeddings(path=path_to_parquet):
    current = time.time()
    df = sinteza.get_all_papers()
    # print(df.head(5))
    print((time.time() - current), 'time from start')
    current = time.time()
    model = stc.load_model(stc.path_to_gte)
    df['title_embed'] = df['title'].apply(model.encode)
    print((time.time() - current))
    # print(df.head(10))
    df.to_parquet(path)

def read_titles_embeddings(path=path_to_parquet):
    return pd.read_parquet(path)

def first_retrieval(df, embedding):
    df['title_dist'] = df['title_embed'].apply(lambda x: stc.cosine_distance(x, embedding))
    # return(df.loc[df['title_rank'] > 0.31])
    return(df.sort_values(by='title_dist')[['link', 'title', 'title_dist']].head(10))

def second_retrieval(df, embedding):
    df['abstract'] = df['link'].apply(sinteza.scrap_abstract)
    model = stc.load_model(stc.path_to_gte)
    df['abstract_embed'] = df['abstract'].apply(model.encode)
    df['abstract_dist'] = df['abstract_embed'].apply(lambda x: stc.cosine_distance(x, embedding))
    # return(df.loc[df['abstract_dist'] > 0.31])
    return(df.sort_values(by='abstract_dist')[['link', 'title', 'title_dist', 'abstract', 'abstract_dist']].head(10))

# def find_k_similar_texts(topic, k):
#     print('get promt embedding')
#     query_embedding = get_embedding(topic)

#     print('vector search')
#     distances, indices = faiss_controller.similarity_search(faiss_controller.experimental_path, query_embedding, k)

#     for i in range(len(indices[0])):
#         try:
#             print(indices[0][i], distances[0][i])
#         except TypeError:
#             print('Entry not found')

if __name__ == "__main__":
    # save_titles_embeddings()

    promt = 'Automating literature search with the help of local embedding models'
    # promt = 'Automating literature search with the help of local embedding models '
    # promt = 'Rastući značaj inovacija u procesu dostizanja održive konkurentne prednosti je u najvećoj meri i uslovio ubrzani razvoj internet marketinga i inicirao sve veću upotrebu društvenih mreža u cilju unapređenja dugoročnog odnosa organizacije i ciljne grupe potrošača. U novoj ekonomiji, odnosno ekonomiji znanja, sam pojam inovacija, pored značajnih unapređenja proizvoda i procesa, sve više podrazumeva organizacione i marketing inovacije, a naročito inovacije u oblasti internet marketinga. Usled navedenih trendova, u ovom naučnom radu, posebna pažnja je usmerena upravo na analizu uticaja koji internet marketing i upotreba internet društvenih mreža imaju na proces unapređenja dvosmerne, odnosno interaktivne komunikacije sa potrošačima. Analizom su obuhvaćeni i brojni primeri realizovanih istraživanja čiji dobijeni rezultati ukazuju da upravo inovacije u oblasti internet marketinga postaju krucijalni faktor direferenciranja i dostizanja održive konkurentne prednosti organizacija, ali i ukazuju kakvo je trenutno stanje u oblasti internet marketinga i upotrebe internet društvenih mreža u Republici Srbiji.'
    
    model = stc.load_model(stc.path_to_gte)
    embedding = model.encode(promt)
    df = read_titles_embeddings()
    df_first = first_retrieval(df, embedding)
    df_second = second_retrieval(df_first, embedding)
    pd.set_option('display.max_colwidth', 20)
    print(df_second)
    # df_second.to_csv('.\\result.csv')
    # print(df_second['link'].to_markdown())

    # df = read_titles_embeddings()
    # print(df.iloc[0]['title_embed'].dtype)


    # title = 'A New Hybrid Descent Algorithm for Large-Scale Nonconvex Optimization and Application to Some Image Restoration Problems'
    # find_k_similar_texts(title, 10)
    # topic = '''Conjugate gradient methods are widely used and attractive for large-scale unconstrained smooth optimization problems, with simple computation, low memory requirements, and interesting theoretical information on the features of curvature. Based on the strongly convergent property of the Dai–Yuan method and attractive numerical performance of the Hestenes–Stiefel method, a new hybrid descent conjugate gradient method is proposed in this paper. The proposed method satisfies the sufficient descent property independent of the accuracy of the line search strategies. Under the standard conditions, the trust region property and the global convergence are established, respectively. Numerical results of 61 problems with 9 large-scale dimensions and 46 ill-conditioned matrix problems reveal that the proposed method is more effective, robust, and reliable than the other methods. Additionally, the hybrid method also demonstrates reliable results for some image restoration problems.'''
    # find_k_similar_texts(topic, 10)
    0

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