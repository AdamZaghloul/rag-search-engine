#!/usr/bin/env python3

import argparse
from lib.semantic_search import *


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Make an instance of the semantic search model and print its details.")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for given text.")
    embed_parser.add_argument("text", type=str, help="Text to embed.")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings.")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a given user query as vectors.")
    embed_query_parser.add_argument("query", type=str, help="Query to embed.")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search.")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

            pass

        case "embed_text":

            try:
                embed_text(args.text)
            except Exception as e:
                print(e)
                return
            
            pass
        
        case "verify_embeddings":

            try:
                verify_embeddings()
            except Exception as e:
                print(e)
                return
            
            pass
        
        case "embedquery":

            try:
                embed_query_text(args.query)
            except Exception as e:
                print(e)
                return
            
            pass

        case "search":

            #try:
            model = SemanticSearch()

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            embeddings = model.load_or_create_embeddings(data["movies"])

            results = model.search(args.query, args.limit)

            for i in range(len(results)):
                print(f"{i+1}.\t{results[i]['title']} (score: {results[i]['score']})")
                print(f"\t{results[i]['description']}")
                print()

            #except Exception as e:
                #print(e)
                #return
            
            pass

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()