import faiss

papers_path = '../data/faiss/papers.faiss'
experimental_path = '../data/faiss/experimental.faiss'

def add_to_faiss(embedding, path):
    index = faiss.read_index(path)
    index.add(embedding)
    print('vector added')
    faiss.write_index(index, path)
    print('faiss saved')


def num_of_vectors(path):
    index = faiss.read_index(path)
    return index.ntotal

if __name__ == "__main__":
    # embedding = get_embedding('The article examines the issue of drug clustering. Initially, k classes are arbitrarily formed and theresulting training sample is pre-processed, then the similarities between the objects of each classare evaluated based on the proximity function and the criterion for evaluating the contribution ofobjects to the formation of their own class. Usually, it is in percentage and is the degree of mutualsimilarity of objects of each class. In the next steps of the algorithm, first, one object is takenfrom the first class, and by adding it to all k classes, the contribution of this object to this classis measured. The object will be left in the class which has the most contribution. This processis repeated several times in a row for all objects of the class. The process is stopped when thelocation of objects does not change and the degree of similarity exceeds the required percentage.As a result, the required clusters are formed.', gai)
    # print(len(embedding))
    print(num_of_vectors(papers_path))
    # index = faiss.IndexFlatL2(768)
    # faiss.write_index(index, experimental_path)