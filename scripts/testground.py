from sentence_transformers import SentenceTransformer

if __name__ == "__main__":    
  # thisdict = {
  #   "brand": "Ford",
  #   "model": "Mustang",
  #   "year": 1964
  # }

  # print(thisdict['brand', 'model'])
  print('start ')
  model = SentenceTransformer('C:\\Misha\\University\\FinalThesis\\Automatic-Literature-Review-with-RAG\\st_models\\gte-multilingual-base', trust_remote_code=True, fix_mistral_regex=True)
  print('loaded')
  embeddings = model.encode('Enhancing Retrieval - Augmented Generation with Graph-Based Retrieval and Generative Modeling')  
  print(type(embeddings[0]))
  0