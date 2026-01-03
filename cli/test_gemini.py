import os
from google import genai
from dotenv import load_dotenv

def main():

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:15]}...")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model='gemini-2.0-flash-001', 
        contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    )

    print(response.text)
    print()
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
    print()

main()