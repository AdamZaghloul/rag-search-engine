from sentence_transformers import SentenceTransformer
import numpy as np
import os, json

class SemanticSearch:
    def __init__(self):

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Text contains only whitespace.")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents):

        self.documents = documents

        doc_strings = []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

        with open("cache/movie_embeddings.npy", "wb") as file:
            np.save(file, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.isfile("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy", "rb") as file:
                self.embeddings = np.load(file)

                if len(self.embeddings) == len(documents):
                    return self.embeddings
                
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = embed_query_text(query)

        similarity_scores = []

        for i in range(len(self.embeddings)):
            similarity_scores.append((cosine_similarity(query_embedding, self.embeddings[i]), self.documents[i]))
        
        similarity_scores.sort(key = lambda x: x[0], reverse=True)

        results = []

        for i in range(limit):
            if i > len(similarity_scores):
                break
            
            dic = {}
            dic["score"] = similarity_scores[i][0]
            dic["title"] = similarity_scores[i][1]["title"]
            dic["description"] = similarity_scores[i][1]["description"]

            results.append(dic)
        
        return results




def verify_model():

    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text):

    model = SemanticSearch()

    embedding = model.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():

    model = SemanticSearch()

    data = None

    with open("data/movies.json", "r") as file:
        data = json.load(file)
    
    embeddings = model.load_or_create_embeddings(data["movies"])

    print(f"Number of docs:   {len(data)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    model = SemanticSearch()

    embedding = model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

    return embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
