import numpy as np
#from google import genai

#import constants
import scrap_kaznu as sp
import faiss_controller as faiss
import sent_trans_controller as stc

#func for robust vector similarity search
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def lang_similarity_exp(path, id, model_path):
    model = stc.load_model(model_path)
    paper_en = sp.get_kaznu_paper(path, id, 'en_US')
    abstract_en = paper_en[1]
    embedding_en = model.encode(abstract_en)
    paper_ru = sp.get_kaznu_paper(path, id, 'ru_RU')
    abstract_ru = paper_ru[1]
    embedding_ru = model.encode(abstract_ru)
    paper_kz = sp.get_kaznu_paper(path, id, 'uk_UA')
    abstract_kz = paper_kz[1]
    embedding_kz = model.encode(abstract_kz)
    embedings = [embedding_en, embedding_ru, embedding_kz]
    print(model.similarity(embedings, embedings))

# def lang_similarity_exp(path, id, client):
#     paper_en = sp.get_kaznu_paper(path, id, 'en_US')
#     abstract_en = paper_en[1]
#     embedding_en = faiss.get_embedding(abstract_en, client)
#     paper_ru = sp.get_kaznu_paper(path, id, 'ru_RU')
#     abstract_ru = paper_ru[1]
#     embedding_ru = faiss.get_embedding(abstract_ru, client)
#     paper_kz = sp.get_kaznu_paper(path, id, 'uk_UA')
#     abstract_kz = paper_kz[1]
#     embedding_kz = faiss.get_embedding(abstract_kz, client)

#     dist_en_ru = cosine_distance(embedding_en, embedding_ru)
#     dist_en_kz = cosine_distance(embedding_en, embedding_kz)
#     dist_ru_kz = cosine_distance(embedding_ru, embedding_kz)

#     print(f"en-ru: {dist_en_ru:.4f}, en-kz: {dist_en_kz:.4f}, ru-kz: {dist_ru_kz:.4f}")

def compare_paper_to_citations(paper_abstract, citatons_abstracts, client):
    paper_embedding = faiss.get_embedding(paper_abstract, client)
    for citation_abstract in citatons_abstracts:
        citation_embedding = faiss.get_embedding(citation_abstract, client)
        print(cosine_distance(paper_embedding, citation_embedding))
    print('a')

if __name__ == "__main__":
    #gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    #paper_abstract = 'The article examines the issue of drug clustering. Initially, k classes are arbitrarily formed and theresulting training sample is pre-processed, then the similarities between the objects of each classare evaluated based on the proximity function and the criterion for evaluating the contribution ofobjects to the formation of their own class. Usually, it is in percentage and is the degree of mutualsimilarity of objects of each class. In the next steps of the algorithm, first, one object is takenfrom the first class, and by adding it to all k classes, the contribution of this object to this classis measured. The object will be left in the class which has the most contribution. This processis repeated several times in a row for all objects of the class. The process is stopped when thelocation of objects does not change and the degree of similarity exceeds the required percentage.As a result, the required clusters are formed.Key words: Clustering, proximity function, degree of similarity of objects, contribution of objectto the class.'
    #citation_abstracts = ['We argue that there are many clustering algorithms, because the notion of "cluster" cannot be precisely defined Clustering is in the eye of the beholder, and as such, researchers have proposed many induction principles and models whose corresponding optimization problem can only be approximately solved by an even larger number of algorithms Therefore, comparing clustering algorithms, must take into account a careful understanding of the inductive principles involved',
    #                      'The problem of clustering a set of points so as to minimize the maximum intercluster distance is studied. An O(kn) approximation algorithm, where n is the number of points and k is the number of clusters, that guarantees solutions with an objective function value within two times the optimal solution value is presented. This approximation algorithm succeeds as long as the set of points satisfies the triangular inequality. We also show that our approximation algorithm is best possible, with respect to the approximation bound, if P ≠ NP.',
    #                      'Organizing data into sensible groupings is one of the most fundamental modes of understanding and learning. As an example, a common scheme of scientific classification puts organisms into a system of ranked taxa: domain, kingdom, phylum, class, etc. Cluster analysis is the formal study of methods and algorithms for grouping, or clustering, objects according to measured or perceived intrinsic characteristics or similarity. Cluster analysis does not use category labels that tag objects with prior identifiers, i.e., class labels. The absence of category information distinguishes data clustering (unsupervised learning) from classification or discriminant analysis (supervised learning). The aim of clustering is to find structure in data and is therefore exploratory in nature. Clustering has a long and rich history in a variety of scientific fields. One of the most popular and simple clustering algorithms, K-means, was first published in 1955. In spite of the fact that K-means was proposed over 50 years ago and thousands of clustering algorithms have been published since then, K-means is still widely used. This speaks to the difficulty in designing a general purpose clustering algorithm and the ill-posed problem of clustering. We provide a brief overview of clustering, summarize well known clustering methods, discuss the major challenges and key issues in designing clustering algorithms, and point out some of the emerging and useful research directions, including semi-supervised clustering, ensemble clustering, simultaneous feature selection during data clustering, and large scale data clustering.',
    #                      'Despite many empirical successes of spectral clustering methods— algorithms that cluster points using eigenvectors of matrices derived from the data—there are several unresolved issues. First. there are a wide variety of algorithms that use the eigenvectors in slightly different ways. Second, many of these algorithms have no proof that they will actually compute a reasonable clustering. In this paper, we present a simple spectral clustering algorithm that can be implemented using a few lines of Matlab. Using tools from matrix perturbation theory, we analyze the algorithm, and give conditions under which it can be expected to do well. We also show surprisingly good experimental results on a number of challenging clustering problems.']
    #compare_paper_to_citations(paper_abstract, citation_abstracts, gai)
    lang_similarity_exp('https://bm.kaznu.kz/index.php/kaznu/', 1608, 'st_models')
    lang_similarity_exp('https://philart.kaznu.kz/index.php/1-FIL/', 4918, 'st_models')
    lang_similarity_exp('https://bulletin-psysoc.kaznu.kz/index.php/1-psy/', 2106, 'st_models')