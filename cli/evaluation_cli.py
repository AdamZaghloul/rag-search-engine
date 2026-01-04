import argparse, json
import lib.hybrid_search as hybrid_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    print(f"k={limit}")
    print()
    data = None
    documents = None

    with open("data/golden_dataset.json", "r") as f:
        data = json.load(f)

    with open("data/movies.json", "r") as f:
        documents = json.load(f)

    searcher = hybrid_search.HybridSearch(documents["movies"])

    for test_case in data["test_cases"]:

        results = searcher.rrf_search(test_case["query"], limit=limit)
        expected_results = test_case["relevant_docs"]
        retrieved_results = []
        relevant_results = []
        total_retrieved = len(results)
        total_relevant = len(expected_results)
        relevant_retrieved = 0

        for result in results:
            retrieved_results.append(result["doc"]["title"])
            if result["doc"]["title"] in expected_results:
                relevant_retrieved += 1
                relevant_results.append(result["doc"]["title"])
        
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        f1 = 2 * (precision * recall) / (precision + recall)
        
        print()
        print(f"- Query: {test_case['query']}")
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Recall@{limit}: {recall:.4f}")
        print(f"\t- F1 Score: {f1:.4f}")
        print(f"\t- Retrieved: {' ,'.join(retrieved_results)}")
        print(f"\t- Relevant: {' ,'.join(relevant_results)}")
        print()



if __name__ == "__main__":
    main()