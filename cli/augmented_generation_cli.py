import argparse, json
import lib.hybrid_search as hybrid_search


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results to consider.")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize the results for a given query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")

    citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + generate answer) including citations to referenced documents."
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results to consider.")

    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + generate answer) to answer a question."
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results to consider.")
    
    args = parser.parse_args()

    query = args.query
    data = None
    with open("data/movies.json") as f:
        data = json.load(f)

    searcher = hybrid_search.HybridSearch(data["movies"])

    results = None

    if args.limit:
        results = searcher.rrf_search(query,limit=args.limit)
    else:
        results = searcher.rrf_search(query)

    results_list = []
    output_list = []
    for i in range(len(results)):
        results_list.append(f"{i+1}. {results[i]['doc']['title']} - {results[i]['doc']['description']}\n")
        output_list.append(f"\t- {results[i]['doc']['title']}")

    match args.command:
        case "rag":
            
            llm_query = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {chr(10).join(results_list)}

                Provide a comprehensive answer that addresses the query:"""
            
            response = hybrid_search.llm_query(llm_query)

            print()
            print(f"Search Results:\n{chr(10).join(output_list)}")
            print()
            print("RAG Response:")
            print(response)

            pass

        case "summarize":

            llm_query = f"""
                Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {chr(10).join(results_list)}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                """
            
            response = hybrid_search.llm_query(llm_query)

            print()
            print(f"Search Results:\n{chr(10).join(output_list)}")
            print()
            print("LLM Summary:")
            print(response)

            pass

        case "citations":

            llm_query = f"""Answer the question or provide information based on the provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Documents:
                {chr(10).join(results_list)}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
            
            response = hybrid_search.llm_query(llm_query)

            print()
            print(f"Search Results:\n{chr(10).join(output_list)}")
            print()
            print("LLM Answer:")
            print(response)

            pass

        case "question":

            llm_query = f"""Answer the user's question based on the provided movies that are available on Hoopla.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Question: {query}

                Documents:
                {chr(10).join(results_list)}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:"""
            
            response = hybrid_search.llm_query(llm_query)

            print()
            print(f"Search Results:\n{chr(10).join(output_list)}")
            print()
            print("Answer:")
            print(response)

            pass

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()