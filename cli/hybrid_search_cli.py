import argparse, json
import lib.hybrid_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize the list of provided scores.")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="Scores to normalize.")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search movies using a weighted keyword and chunked semantic search.")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=0.5, help="Optional .")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search movies using a weighted keyword and chunked semantic search.")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, nargs='?', default=60, help="Optional .")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print(args.scores)
            normalized = lib.hybrid_search.normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
            
            pass

        case "weighted-search":
            
            with open("data/movies.json", "r") as file:
                data = json.load(file)
            
            model = lib.hybrid_search.HybridSearch(data["movies"])

            results = model.weighted_search(args.query, args.alpha, args.limit)

            count = 1
            for res in results:
                print(f"{count}.  {res['doc']['title']}")
                print(f"    Hybrid Score: {res['hybrid_score']:.4f}")
                print(f"    BM25: {res['keyword_score']}, Semantic: {res['semantic_score']}")
                print(f"    {res['doc']['description'][:100]}...")
                print()
                count += 1
            
            pass

        case "rrf-search":
            with open("data/movies.json", "r") as file:
                data = json.load(file)
            
            model = lib.hybrid_search.HybridSearch(data["movies"])

            query = args.query

            match args.enhance:
                case "spell":
                    llm_query = f"""Fix any spelling errors in this movie search query.

                        Only correct obvious typos. Don't change correctly spelled words.

                        Query: "{query}"

                        If no errors, return the original query.
                        Corrected:"""
                    
                    query = lib.hybrid_search.llm_query(llm_query)

                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

                    pass
                case "rewrite":
                    llm_query = f"""Rewrite this movie search query to be more specific and searchable.

                        Original: "{query}"

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep it concise (under 10 words)
                        - It should be a google style search query that's very specific
                        - Don't use boolean logic

                        Examples:

                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                        Rewritten query:"""
                    
                    query = lib.hybrid_search.llm_query(llm_query)

                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

                    pass

                case "expand":
                    llm_query = f"""Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Examples:

                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        Query: "{query}"
                        """
                    
                    query = lib.hybrid_search.llm_query(llm_query)

                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

                    pass

                case _:
                    pass


            results = model.rrf_search(query, args.k, args.limit)

            count = 1
            for res in results:
                print(f"{count}.\t{res['doc']['title']}")
                print(f"\t\tRRF Score: {res['rrf_score']:.4f}")
                print(f"\t\tBM25 Rank: {res['keyword_rank']}, Semantic Rank: {res['semantic_rank']}")
                print(f"\t\t{res['doc']['description'][:100]}...")
                print()
                count += 1
            
            pass
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()