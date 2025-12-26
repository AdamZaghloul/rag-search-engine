import json, pickle, os, math
from keyword_search_cli import tokenize_text
from collections import Counter, defaultdict

class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

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
    
    def load(self):

        if not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl"):
            raise Exception("Pickle files for index and docmap not found.")
            
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)
    
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