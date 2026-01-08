from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
import json

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []

        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path):
        image = Image.open(path)
        embedding = self.model.encode([image])
        return embedding[0]
    def search_with_image(self, path, limit=5):
        image = Image.open(path)
        image_embedding = self.model.encode([image])
        results = []

        for i in range(len(self.embeddings)):
            dic = {}
            dic["id"] = self.documents[i]['id']
            dic["title"] = self.documents[i]['title']
            dic["description"] = self.documents[i]['description']
            dic["score"] = float(cosine_similarity(image_embedding, self.embeddings[i]))
            results.append(dic)
        
        sorted_results = sorted(
            results,
            key=lambda item: item["score"],
            reverse=True)
        
        return_len = limit

        if limit > len(sorted_results):
            return_len = len(sorted_results)
            
        return sorted_results[:return_len]

def verify_image_embedding(path):
    search = MultimodalSearch()
    embedding = search.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path, limit=5):

    data = None
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    
    search = MultimodalSearch(data["movies"])

    return search.search_with_image(path, limit)