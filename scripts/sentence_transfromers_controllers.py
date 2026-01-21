from sentence_transformers import SentenceTransformer
import time

def save_model(model_name):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save('st_models')

def load_model(model_path):
    return SentenceTransformer(model_path, trust_remote_code=True)

# save_model("Alibaba-NLP/gte-multilingual-base")

if __name__ == "__main__":
    start = time.time()
    print(time.time() - start)
    model = load_model('st_models')
    print(time.time() - start)
    texts  = [
        'The article examines the experience of describing the syntactic structure of Turkic languages from the perspective of formal grammar and based on modern annotation models. Syntactic annotation is recognized as an important tool for formally describing a language’s grammatical system and enabling its automatic processing. Relying on projects such as Universal Dependencies (UD), MaTT (Multilingual Aligned Treebank of Turkic), and the Kazakh Dependency Treebank (KazDT), the study describes morphological and syntactic features characteristic of Turkic languages.',
        'В статье рассматривается опыт описания синтаксической структуры тюркских языков с точки зрения формальной грамматики и на основе современных аннотационных моделей. Синтаксическая аннотация признаётся важным инструментом, позволяющим формально описать грамматическую систему языка и обеспечивающим возможность её автоматической обработки. В ходе исследования, опираясь на проекты «Universal Dependencies» (UD), «MaTT» (Multilingual Aligned Treebank of Turkic) и «Kazakh Dependency Treebank» (KazDT), были описаны морфологические и синтаксические особенности, характерные для тюркских языков.',
        'Мақалада түркі тілдерінің синтаксистік құрылымын формалды грамматика тұрғысынан және заманауи аннотациялық модельдер негізінде сипаттаудың тәжірибесі қарастырылады. Синтаксистік аннотация тілдің грамматикалық жүйесін формалды түрде сипаттайтын және оны автоматты өңдеуге мүмкіндік беретін маңызды құрал ретінде танылады. Зерттеу барысында «Universal Dependencies» (UD), «MaTT» (Multilingual Aligned Treebank of Turkic) және «Kazakh Dependency Treebank» (KazDT) сияқты жобаларға сүйеніп, түркі тілдеріне тән морфологиялық және синтаксистік ерекшеліктер сипатталды.'
    ]
    embeddings = model.encode(texts)
    print(time.time() - start)
    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    print(time.time() - start)
