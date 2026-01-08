import argparse, mimetypes
import lib.hybrid_search as hybrid_search
import google.genai.types as types

def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")

    parser.add_argument("--image", type=str, help="Path to image file.")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image.")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    image_contents = None

    with open(args.image, "rb") as f:
        image_contents = f.read()
    
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary"""
    
    parts = [
        system_prompt,
        types.Part.from_bytes(data=image_contents, mime_type=mime),
        args.query.strip()
    ]

    response = hybrid_search.llm_query(parts)

    print(f"Rewritten query: {response.strip()}")

if __name__ == "__main__":
    main()