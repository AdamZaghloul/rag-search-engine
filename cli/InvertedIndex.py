import json, pickle, os
from keyword_search_cli import tokenize_text

class InvertedIndex:
    index = {}
    docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        for token in tokens:
            if token in self.index:
                self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]

    def get_documents(self,term):
        return sorted(self.index[term.lower()])
    
    def build(self):
        with open("data/movies.json", "r") as file:
            data = json.load(file)

            for movie in data["movies"]:
                self.docmap[movie["id"]] = movie
                self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
    
    def save(self):

        if not os.path.isdir("cache"):
            os.mkdir("cache")
        
        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
    
    def load(self):

        if not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl"):
            raise Exception("Pickle files for index and docmap not found.")
            
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)