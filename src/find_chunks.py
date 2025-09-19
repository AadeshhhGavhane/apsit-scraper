import os
import sys
from typing import List
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")

assert PINECONE_API_KEY, "❌ Missing PINECONE_API_KEY"
assert PINECONE_INDEX, "❌ Missing PINECONE_INDEX"
assert PINECONE_INDEX_HOST, "❌ Missing PINECONE_INDEX_HOST"
assert PINECONE_NAMESPACE, "❌ Missing PINECONE_NAMESPACE"
assert MISTRALAI_API_KEY, "❌ Missing MISTRALAI_API_KEY"


def find_chunks_by_professor_name(professor_name: str, k: int = 5):
    # Initialize Pinecone client (ensures connectivity)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ = pc.Index(PINECONE_INDEX, host=PINECONE_INDEX_HOST)

    # Embeddings must match what was used during ingestion
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRALAI_API_KEY)

    # Create vector store handle
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    # Simple query string; filter narrows to this professor
    query = f"Information about {professor_name}"

    # Perform similarity search with a Pinecone filter on metadata
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        filter={"professor_name": professor_name},
    )

    if not results:
        print("No chunks found.")
        return

    for i, doc in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Rank: {i}")
        print("Metadata:")
        for key, value in (doc.metadata or {}).items():
            print(f"  {key}: {value}")
        print("Content:")
        print(doc.page_content[:800])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/find_chunks.py <Professor Name> [k]")
        sys.exit(1)
    name = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    find_chunks_by_professor_name(name, k=top_k) 