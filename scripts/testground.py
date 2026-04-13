from sentence_transformers import SentenceTransformer
import os
import shutil
from pathlib import Path

if __name__ == "__main__":
  

  cache_path = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "gte_hyphen_multilingual_hyphen_base"

  if cache_path.exists():
    shutil.rmtree(cache_path)
    print(f"Removed cached model at {cache_path}")

  0