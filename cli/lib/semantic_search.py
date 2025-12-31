from sentence_transformers import SentenceTransformer
from constants import *
import numpy as np
import os, json, re

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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):

        self.documents = documents

        all_chunks = []
        chunk_data = []

        for doc in documents:

            self.document_map[doc["id"]] = doc
            
            if "description" not in doc:
                continue
            elif doc["description"] == None or doc["description"] == "":
                continue

            chunks = semantic_chunk(doc["description"], 4, 1)
            all_chunks.extend(chunks)

            for i in range(len(chunks)):
                dic = {}
                dic["movie_idx"] = documents.index(doc)
                dic["chunk_idx"] = i
                dic["total_chunks"] = len(chunks)
                chunk_data.append(dic)


        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_data

        with open("cache/chunk_embeddings.npy", "wb") as file:
            np.save(file, self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as file:
            json.dump({"chunks": chunk_data, "total_chunks": len(all_chunks)}, file, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents

        for doc in documents:

            self.document_map[doc["id"]] = doc
        
        if not (os.path.isfile("cache/chunk_embeddings.npy") and os.path.isfile("cache/chunk_metadata.json")):
            return self.build_chunk_embeddings(documents)

        with open("cache/chunk_embeddings.npy", "rb") as file:
            self.chunk_embeddings = np.load(file)
        
        with open("cache/chunk_metadata.json", "r") as file:
            self.chunk_metadata = json.load(file)["chunks"]
        
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = embed_query_text(query)
        chunk_scores = []

        for i in range(len(self.chunk_embeddings)):
            dic = {}
            cosine = cosine_similarity(embedded_query, self.chunk_embeddings[i])
            dic["chunk_idx"] = self.chunk_metadata[i]["chunk_idx"]
            dic["movie_idx"] = self.chunk_metadata[i]["movie_idx"]
            dic["score"] = cosine
            chunk_scores.append(dic)
        
        movie_scores = {}

        for chunk_score in chunk_scores:
            if chunk_score["movie_idx"] not in movie_scores:
                movie_scores[chunk_score["movie_idx"]] = chunk_score["score"]
            else:
                if chunk_score["score"] > movie_scores[chunk_score["movie_idx"]]:
                    movie_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        chunk_scores.sort(key = lambda x: x["score"], reverse=True)

        results = []
        result_len = limit
        if len(chunk_scores) < limit:
            result_len = len(chunk_scores)

        for i in range(result_len):
            dic = {}
            dic["id"] = chunk_scores[i]["movie_idx"]
            dic["title"] = self.documents[dic["id"]]["title"]
            dic["document"] = self.documents[dic["id"]]["description"][:100]
            dic["score"] = round(chunk_scores[i]["score"], SCORE_PRECISION)
            dic["metadata"] = self.chunk_metadata[dic["id"]]
            results.append(dic)
        
        return results
    
def semantic_chunk(text, chunk_size, overlap):
    
    text = text.strip()
    if text == "" or text == None:
        return []

    words = re.split(r"(?<=[.!?])\s+", text)

    if len(words) == 1 and not words[0].endswith((".", "!", "?")):
        return [words[0].strip()]
    
    chunks = []

    for i in range(0, len(words), chunk_size-overlap):

        new_chunk = ""

        if i + chunk_size - overlap >= len(words) - 1:
            new_chunk = " ".join(words[i:]).strip()
        else:
            new_chunk = " ".join(words[i:i + chunk_size]).strip()

        if new_chunk != "" and new_chunk != None:
            chunks.append(new_chunk)
    
    return chunks
