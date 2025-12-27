import json, pickle, os, math
from keyword_search_cli import tokenize_text
from collections import Counter, defaultdict
from constants import *

class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self,term):
        return sorted(self.index[term.lower()])
    
    def build(self):
        with open("data/movies.json", "r") as file:
            data = json.load(file)

            for movie in data["movies"]:
                self.docmap[movie["id"]] = movie
                self.term_frequencies[movie["id"]] = Counter()
                self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
    
    def save(self):

        if not os.path.isdir("cache"):
            os.mkdir("cache")
        
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)
        
        with open("cache/doc_lengths.pkl", "wb") as file:
            pickle.dump(self.doc_lengths, file)
    
    def load(self):

        if not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl"):
            raise Exception("Pickle files for index and docmap not found.")
            
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)
        
        with open("cache/doc_lengths.pkl", "rb") as file:
            self.doc_lengths = pickle.load(file)
    
    def get_tf(self, doc_id, term):
        token = tokenize_text(term)

        if len(token) > 1:
            raise Exception("Too Many Tokens")
        
        if doc_id in self.term_frequencies:
            freq = self.term_frequencies[doc_id]
            if token[0] in freq:
                return freq[token[0]]
        
        return 0
    
    def get_idf(self, term):

        token = tokenize_text(term)

        if len(token) > 1:
            raise Exception("Too Many Tokens")
        
        token = token[0]

        total_doc_count = len(self.docmap)
        term_match_doc_count = 0

        if token in self.index:
            term_match_doc_count = len(self.index[token])

        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

        return idf
    
    def get_bm25_idf(self, term: str) -> float:

        token = tokenize_text(term)

        if len(token) > 1:
            raise Exception("Too Many Tokens")
        
        token = token[0]

        N = len(self.docmap)
        df = 0

        if token in self.index:
            df = len(self.index[token])

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())

        bm25tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return bm25tf
    
    def __get_avg_doc_length(self) -> float:

        if len(self.doc_lengths) == 0:
            return 0.0
        
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def bm25(self, doc_id, term):

        bm25tf = self.get_bm25_tf(doc_id, term)
        bm25idf = self.get_bm25_idf(term)

        return bm25tf * bm25idf
    
    def bm25_search(self, query, limit=5):

        tokens = tokenize_text(query)
        scores = {}

        for doc_id in self.docmap:
            doc_score = 0
            for token in tokens:
                doc_score += self.bm25(doc_id, token)
            
            scores[doc_id] = doc_score
        
        scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

        return dict(list(scores.items())[:limit])