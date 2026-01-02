import os

from InvertedIndex import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("data/movies.json"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        keyword_ids = list(bm25_results.keys())
        keyword_scores = list(bm25_results.values())
        keyword_scores = normalize_scores(keyword_scores)

        semantic_ids = []
        semantic_scores = []

        for res in semantic_results:
            semantic_ids.append(res["id"])
            semantic_scores.append(res["score"])
            
        semantic_scores = normalize_scores(semantic_scores)

        score_dict = {}

        for i in range(len(keyword_ids)):
            dic = {}
            dic["doc"] = self.idx.docmap[keyword_ids[i]]
            dic["keyword_score"] = keyword_scores[i]
            dic["semantic_score"] = 0

            score_dict[keyword_ids[i]] = dic

        for i in range(len(semantic_ids)):
            movie_id = self.documents[semantic_ids[i]]["id"]
            if movie_id in score_dict:
                score_dict[movie_id]["semantic_score"] = semantic_scores[i]
                score_dict[movie_id]["hybrid_score"] = hybrid_score(score_dict[movie_id]["keyword_score"] ,semantic_scores[i], alpha)
            else:
                score_dict[movie_id]["doc"] = self.idx.docmap[movie_id]
                score_dict[movie_id]["semantic_score"] = semantic_scores[i]
                score_dict[movie_id]["keyword_score"] = 0
                score_dict[movie_id]["hybrid_score"] = hybrid_score(0 ,semantic_scores[i], alpha)
        
        sorted_results = sorted(
            score_dict.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True)

        result_len = limit
        
        if limit > len(sorted_results):
            result_len = len(sorted_results)

        return sorted_results[:result_len]    
        

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    dif = 0

    if min_score == max_score:
        dif = 0
    else:
        dif = max_score-min_score

    for i in range(len(scores)):
        if dif == 0:
            scores[i] = 1
        else:
            scores[i] = (scores[i] - min_score)/(dif)

    return scores

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)