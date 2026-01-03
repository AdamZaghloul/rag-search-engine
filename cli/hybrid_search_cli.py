import argparse, json, time
import lib.hybrid_search
from sentence_transformers import CrossEncoder


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
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query LLM rerank method",
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

            limit = args.limit
            
            match args.rerank_method:
                case "individual" | "batch" | "cross_encoder":
                    limit *= 5
                    pass
                case _:
                    pass

            results = model.rrf_search(query, args.k, args.limit)
            final_results = []

            match args.rerank_method:
                case "individual":

                    for i in range(len(results)):
                        llm_query = f"""Rate how well this movie matches the search query.

                            Query: "{query}"
                            Movie: {results[i]["doc"].get("title", "")} - {results[i]["doc"].get("description", "")}

                            Consider:
                            - Direct relevance to query
                            - User intent (what they're looking for)
                            - Content appropriateness

                            Rate 0-10 (10 = perfect match).
                            Give me ONLY the number in your response, no other text or explanation.

                            Score:"""
                        
                        results[i]["rerank_score"] = int(lib.hybrid_search.llm_query(llm_query))
                        time.sleep(3)
                    
                    sorted_results = sorted(
                        results,
                        key=lambda item: item["rerank_score"],
                        reverse=True)
                    
                    return_len = args.limit

                    if limit > len(sorted_results):
                        return_len = len(sorted_results)

                    final_results = sorted_results[:return_len]

                    print(f"Reranking top {args.limit} results using {args.rerank_method} method...")

                    pass
                case "batch":
                    doc_list_str = ""
                    for i in range(len(results)):
                        doc_list_str += f"ID: {results[i]['doc'].get('id', '')} - {results[i]['doc'].get('title', '')} - {results[i]['doc'].get('description', '')}\n\n"

                    llm_query = f"""Rank these movies by relevance to the search query.

                        Query: "{query}"

                        Movies:
                        {doc_list_str}

                        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                        [75, 12, 34, 2, 1]
                        """

                    batch_order = json.loads(lib.hybrid_search.llm_query(llm_query))
                    result_len = 0

                    for i in range(len(batch_order)):
                        for j in range(len(results)):
                            if batch_order[i] == results[j]['doc']['id']:
                                results[j]['rerank_score'] = i+1
                                final_results.append(results[j])
                                result_len += 1
                                break
                        if result_len >= args.limit:
                            break
                    print(f"Reranking top {args.limit} results using {args.rerank_method} method...")
                case "cross_encoder":
                    pairs = []

                    for i in range(len(results)):
                        pairs.append([query, f"{results[i]['doc'].get('title', '')} - {results[i]['doc'].get('description', '')}"])

                    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                    scores = cross_encoder.predict(pairs)

                    for i in range(len(scores)):
                        results[i]['rerank_score'] = scores[i]

                    sorted_results = sorted(
                        results,
                        key=lambda item: item["rerank_score"],
                        reverse=True)
                    
                    return_len = args.limit

                    if limit > len(sorted_results):
                        return_len = len(sorted_results)

                    final_results = sorted_results[:return_len]

                    print(f"Reranking top {args.limit} results using {args.rerank_method} method...")

                    pass
                case _:
                    final_results = results
                    pass

            count = 1
            for res in final_results:
                print(f"{count}.\t{res['doc']['title']}")

                match args.rerank_method:
                    case "individual" | "batch" | "cross_encoder":
                        print(f"\t\tRerank Score: {res['rerank_score']}/10")
                    case _:
                        pass

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