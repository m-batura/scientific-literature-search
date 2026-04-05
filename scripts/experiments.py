import numpy as np
#from google import genai

#import constants
import scrap_kaznu as sp
import faiss_controller as faiss
import sent_trans_controller as stc

#func for robust vector similarity search
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def similarity_experiment():
    data = [
        (1, 'https://doi.org/10.15308/Sinteza-2025-3-9', 'Enhancing Retrieval - Augmented Generation with Graph-Based Retrieval and Generative Modeling', 'This paper presents the design and implementation of a robust RetrievalAugmented Generation (RAG) system that integrates advanced retrieval, ranking, and generative techniques to address knowledge-intensive tasks. The system combines dense retrieval using ChromaDB, metadata-driven keyword extraction with YAKE and KMedoids algorithm for clustering keywords, graph-based retrieval leveraging PageRank, and cross-encoder re-ranking to deliver precise and contextually relevant results. These retrieval outputs are synthesized into high-quality conversational responses using Hugging Face models and Google API. A modular pipeline ensures scalability, seamlessly integrating various retrieval and generative components. Evaluation results demonstrate high retrieval precision, improved recall through graph-based methods, and enhanced response quality through structured prompt engineering. This work highlights the effectiveness of combining diverse techniques in RAG systems, offering a foundation for scalable, reliable, and context-aware applications in domains such as customer support, education, and research.'),
        (1, 'https://doi.org/10.48550/arXiv.2005.11401', 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', 'Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.'),
        (2, 'https://doi.org/10.15308/Sinteza-2025-78-85', 'Leveraging LLMS for Automatic Forum Scraper Generation', 'Web forums contain valuable user-generated content (UGC), but crawling them presents a challenging task due to the differences between forum technologies and structures. This paper proposes a general approach that uses Large Language Models (LLMs) to automatically detect the forum technology (e.g., phpBB, vBulletin, SMF, Discuz!) and generate a web scraper for that forum’s layout, structure, and pagination. LLM first identifies the platform of a given forum by analysing its HTML patterns after it generates code to efficiently collect available posts and threads that are publicly available and don’t require user registration. Several state-of-the-art LLMs are evaluated (GPT-4, Claude 2, and Mistral 7B) for this task, comparing their speed, accuracy, and reliability in generating functional scraping code. A proof-of-concept functionality was demonstrated on a chosen phpBB forum technology by crawling its content with LLM-generated Python code. Experimental results show that the LLM-generated scrapers can successfully retrieve forum posts with high accuracy, matching manually coded crawlers while adapting automatically to different forum structures. The findings suggest that LLMs can significantly improve forum data collection, avoiding manual per-site adjustments and reducing duplicate content in incremental crawls.'),
        (2, '', '', ''),
    ]

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
    # lang_similarity_exp('https://bm.kaznu.kz/index.php/kaznu/', 1608, 'st_models')
    # lang_similarity_exp('https://philart.kaznu.kz/index.php/1-FIL/', 4918, 'st_models')
    # lang_similarity_exp('https://bulletin-psysoc.kaznu.kz/index.php/1-psy/', 2106, 'st_models')
    0