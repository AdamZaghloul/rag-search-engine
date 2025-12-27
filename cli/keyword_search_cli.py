#!/usr/bin/env python3

import argparse, json, string, InvertedIndex
from nltk.stem import PorterStemmer
from constants import *

def get_stopwords():
    with open("data/stopwords.txt", "r") as file:
        return file.read().splitlines()
    
def tokenize_text(text):
    stopwords = get_stopwords()
    stemmer = PorterStemmer()

    keyword_tokens_temp = text.lower().translate(str.maketrans("", "", string.punctuation)).split()

    filtered = [w for w in keyword_tokens_temp if w not in stopwords]

    keyword_tokens = [stemmer.stem(w) for w in filtered]

    return keyword_tokens

def find_movies(keyword, index):

    keyword_tokens = tokenize_text(keyword)

    results = []
    seen = set()

    for keyword in keyword_tokens:
        if keyword not in index.index:
            continue

        for doc_id in index.get_documents(keyword):
            if doc_id in seen:
                continue

            seen.add(doc_id)
            results.append(doc_id)

            if len(results) >= 5:
                return results
    
    return results

    
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index for movie searches")

    tf_parser = subparsers.add_parser("tf", help="See the term frequency for a given doc_id and a given term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="See the inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="See the tf-idf score for a given term in a given document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="See the BM25 inverse document frequency for a given term")
    bm25idf_parser.add_argument("term", type=str, help="Search term")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="See the BM25 term frequency for a given doc_id and a given term")
    bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25tf_parser.add_argument("term", type=str, help="Search term")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            index = InvertedIndex.InvertedIndex()

            try:
                index.load()
            except Exception as e:
                print(e)
                return
            
            movies = find_movies(args.query, index)
            
            for movie in movies:
                print(f"{movie}. {index.docmap[movie]['title']}")

            pass
        
        case "build":
            index = InvertedIndex.InvertedIndex()
            index.build()
            index.save()

            pass

        case "tf":

            index = InvertedIndex.InvertedIndex()

            try:
                index.load()
                print(index.get_tf(args.doc_id, args.term))
            except Exception as e:
                print(e)
                return
            
            pass

        case "idf":

            index = InvertedIndex.InvertedIndex()

            try:
                index.load()
                
                idf = index.get_idf(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

            except Exception as e:
                print(e)
                return
            
            pass

        case "tfidf":

            index = InvertedIndex.InvertedIndex()

            try:
                index.load()

                tf = index.get_tf(args.doc_id, args.term)
                idf = index.get_idf(args.term)

                tf_idf = tf * idf
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

            except Exception as e:
                print(e)
                return
            
            pass

        case "bm25idf":
            index = InvertedIndex.InvertedIndex()

            try:
                index.load()
                
                bm25idf = index.get_bm25_idf(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

            except Exception as e:
                print(e)
                return
            
            pass

        case "bm25search":
            index = InvertedIndex.InvertedIndex()

            try:
                index.load()
                results = index.bm25_search(args.query)
                
                i = 1
                for key in results:
                    print(f"{i}. ({key}) {index.docmap[key]['title']} - {results[key]:.2f}")
                    i += 1

            except Exception as e:
                print(e)
                return
            pass

        case "bm25tf":

            index = InvertedIndex.InvertedIndex()

            try:
                index.load()
                bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except Exception as e:
                print(e)
                return
            pass

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()