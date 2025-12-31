#!/usr/bin/env python3

import argparse, re
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
    search_parser.add_argument("--limit", type=int, nargs='?', default=200, help="Optional maximum number of results.")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text into strings of a given size.")
    chunk_parser.add_argument("text", type=str, help="Text to chunk.")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=5, help="Optional number of words to include in a chunk.")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Optional number of words to overlap with preceeding chunk.")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a text into strings of a given number of sentences.")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk.")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Optional number of sentences to include in a chunk.")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Optional number of sentences to overlap with preceeding chunk.")
    
    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for chunks of movie data.")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic search.")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")

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

            model = SemanticSearch()

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            embeddings = model.load_or_create_embeddings(data["movies"])

            results = model.search(args.query, args.limit)

            for i in range(len(results)):
                print(f"{i+1}.\t{results[i]['title']} (score: {results[i]['score']})")
                print(f"\t{results[i]['description']}")
                print()
            
            pass

        case "chunk":

            words = args.text.split()
            chunk_size = args.chunk_size
            overlap = args.overlap
            count = 1

            print(f"Chunking {len(args.text)} characters")

            for i in range(0, len(words), chunk_size):
                start = 0
                end = 0
                if i == 0:
                    start = 0
                    end = i + chunk_size
                else:
                    start = i - overlap
                    end = i + chunk_size - overlap

                if i + chunk_size - overlap >= len(words):
                    print(f"{count}.", " ".join(words[start:]))
                    break
                print(f"{count}.", " ".join(words[start:end]))
                count += 1
            
            pass

        case "semantic_chunk":

            words = re.split(r"(?<=[.!?])\s+", args.text)
            chunk_size = args.max_chunk_size
            overlap = args.overlap
            count = 1

            print(f"Semantically chunking {len(args.text)} characters")

            for i in range(0, len(words), chunk_size-overlap):

                if i + chunk_size - overlap >= len(words) - 1:
                    print(f"{count}.", " ".join(words[i:]))
                    break

                print(f"{count}.", " ".join(words[i:i + chunk_size]))
                count += 1
            
            pass

        case "embed_chunks":

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            chunked_search = ChunkedSemanticSearch()

            embeddings = chunked_search.load_or_create_chunk_embeddings(data["movies"])

            print(f"Generated {len(embeddings)} chunked embeddings")

            pass

        case "search_chunked":
            model = ChunkedSemanticSearch()

            data = {}
            with open("data/movies.json", "r") as file:
                data = json.load(file)

            embeddings = model.load_or_create_chunk_embeddings(data["movies"])

            results = model.search_chunks(args.query, args.limit)

            i = 1
            for result in results:
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
                i += 1
            
            pass

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()