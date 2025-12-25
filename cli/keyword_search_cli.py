#!/usr/bin/env python3

import argparse, json, string, InvertedIndex
from nltk.stem import PorterStemmer

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
    search_parser = subparsers.add_parser("build", help="Build the inverted index for movie searches")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()