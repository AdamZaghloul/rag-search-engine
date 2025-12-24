#!/usr/bin/env python3

import argparse, json, string
from nltk.stem import PorterStemmer

def get_stopwords():
    with open("data/stopwords.txt", "r") as file:
        return file.read().splitlines()

def find_movies(keyword):
    movies =[]
    found_token = False

    stopwords = get_stopwords()

    with open("data/movies.json", "r") as file:
        data = json.load(file)
        stemmer = PorterStemmer()

        keyword_tokens_temp = keyword.lower().translate(str.maketrans("", "", string.punctuation)).split()
        keyword_tokens = []

        for keyword in keyword_tokens_temp:
            if keyword not in stopwords:
                keyword_tokens.append(stemmer.stem(keyword))

        for movie in data["movies"]:
            movie_tokens = movie["title"].lower().translate(str.maketrans("", "", string.punctuation)).split()
            for keyword_token in keyword_tokens:
                for movie_token in movie_tokens:
                    if keyword_token in movie_token and keyword_token not in stopwords:
                        movies.append(movie["title"])
                        found_token = True
                        break
                if found_token:
                    found_token = False
                    break
    
    return movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            movies = find_movies(args.query)
            
            i = 1
            for movie in movies:
                print(f"{i}. {movie}")
                i += 1

            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()