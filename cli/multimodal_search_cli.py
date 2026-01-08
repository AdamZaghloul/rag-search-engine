import argparse, lib.multimodal_search

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Check if a given image at path i embedded correctly.")
    verify_parser.add_argument("path", type=str, help="Path to image to verify.")

    image_search_parser = subparsers.add_parser("image_search", help="Search movies based on the provided image.")
    image_search_parser.add_argument("path", type=str, help="Path to image to search for.")
    image_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Optional maximum number of results.")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":

            lib.multimodal_search.verify_image_embedding(args.path)
            pass

        case "image_search":

            results = lib.multimodal_search.image_search_command(args.path, args.limit)

            count = 1
            for res in results:
                print(f"{count}.\t{res['title']} (similarity: {res['score']:.3f})")
                print(f"\t{res['description'][:100]}")
                print()
                count += 1
            pass

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()