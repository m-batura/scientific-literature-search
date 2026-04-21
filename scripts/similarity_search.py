import pandas as pd
import scrap_sinteza as sinteza
import sent_trans_controller as stc
import time

path_to_json = '.\\titles.json'
path_to_parquet = '.\\titles.parquet'

def save_titles_embeddings(path=path_to_parquet):
    current = time.time()
    df = sinteza.get_all_papers()
    print((time.time() - current), 'time from start')
    current = time.time()
    model = stc.load_model(stc.path_to_gte)
    df['title_embed'] = df['title'].apply(model.encode)
    print((time.time() - current))
    print(len(df.index))
    df.to_parquet(path)

def read_titles_embeddings(path=path_to_parquet):
    return pd.read_parquet(path)

# first stage of retrieval
def first_retrieval(df, embedding):
    df['title_dist'] = df['title_embed'].apply(lambda x: stc.cosine_distance(x, embedding))
    return(df.sort_values(by='title_dist', ascending=False)[['link', 'title', 'title_dist']].head(10))

# second stage of retrieval
def second_retrieval(df, embedding):
    df['abstract'] = df['link'].apply(sinteza.scrap_abstract)
    model = stc.load_model(stc.path_to_gte)
    df['abstract_embed'] = df['abstract'].apply(model.encode)
    df['abstract_dist'] = df['abstract_embed'].apply(lambda x: stc.cosine_distance(x, embedding))
    # return(df.loc[df['abstract_dist'] > 0.31])
    return(df.sort_values(by='abstract_dist', ascending=False)[['link', 'title', 'title_dist', 'abstract', 'abstract_dist']].head(10))

if __name__ == "__main__":
    save_titles_embeddings()

    # input = 'This study examines Retrieval-Augmented Generation (RAG) in large language models (LLMs) and their significant application for undertaking systematic literature reviews (SLRs). RAG-based LLMs can potentially automate tasks like data extraction, summarization, and trend identification. However, while LLMs are exceptionally proficient in generating human-like text and interpreting complex linguistic nuances, their dependence on static, pre-trained knowledge can result in inaccuracies and hallucinations. RAG mitigates these limitations by integrating LLMs’ generative capabilities with the precision of real-time information retrieval. We review in detail the three key processes of the RAG framework—retrieval, augmentation, and generation. We then discuss applications of RAG-based LLMs to SLR automation and highlight future research topics, including integration of domain-specific LLMs, multimodal data processing and generation, and utilization of multiple retrieval sources. We propose a framework of RAG-based LLMs for automating SRLs, which covers four stages of SLR process: literature search, literature screening, data extraction, and information synthesis. Future research aims to optimize the interaction between LLM selection, training strategies, RAG techniques, and prompt engineering to implement the proposed framework, with particular emphasis on the retrieval of information from individual scientific papers and the integration of these data to produce outputs addressing various aspects such as current status, existing gaps, and emerging trends.'
    input = 'This paper presents the design and implementation of a robust RetrievalAugmented Generation (RAG) system that integrates advanced retrieval, ranking, and generative techniques to address knowledge-intensive tasks. The system combines dense retrieval using ChromaDB, metadata-driven keyword extraction with YAKE and KMedoids algorithm for clustering keywords, graph-based retrieval leveraging PageRank, and cross-encoder re-ranking to deliver precise and contextually relevant results. These retrieval outputs are synthesized into high-quality conversational responses using Hugging Face models and Google API. A modular pipeline ensures scalability, seamlessly integrating various retrieval and generative components. Evaluation results demonstrate high retrieval precision, improved recall through graph-based methods, and enhanced response quality through structured prompt engineering. This work highlights the effectiveness of combining diverse techniques in RAG systems, offering a foundation for scalable, reliable, and context-aware applications in domains such as customer support, education, and research.'

    model = stc.load_model(stc.path_to_gte)
    embedding = model.encode(input)
    df = read_titles_embeddings()
    df_first = first_retrieval(df, embedding)
    df_second = second_retrieval(df_first, embedding)
    pd.set_option('display.max_colwidth', 20)
    print(df_second)
    df_second.to_csv('sim_search_result.csv')
    0