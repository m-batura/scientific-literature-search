import numpy as np
from google import genai
from google.genai import types

import constants

gai = genai.Client(api_key=constants.GEMINI_API_KEY)

def get_embedding(text_to_embed):
    result = gai.models.embed_content(
        model="text-embedding-004",
        contents=text_to_embed,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    print('received embedding')
    raw_embedding = result.embeddings[0].values
    return np.array(raw_embedding, dtype='float32').reshape(1, -1)