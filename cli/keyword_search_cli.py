#!/usr/bin/env python3

import argparse, json

def find_movies(keyword):
    movies =[]
    with open("data/movies.json", "r") as file:
        data = json.load(file)

        for movie in data["movies"]:
            if keyword.lower() in movie["title"].lower():
                movies.append(movie["title"])

    
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