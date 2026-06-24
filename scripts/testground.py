from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained('.\\st_models\\gte-multilingual-base')
  print(type(tokenizer))
  0