import os
import re
import uuid
import time
import requests
from typing import List
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_RPM = int(os.getenv("EMBED_RPM", "60"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))

assert PINECONE_API_KEY, "❌ Missing PINECONE_API_KEY"
assert PINECONE_INDEX, "❌ Missing PINECONE_INDEX"
assert PINECONE_INDEX_HOST, "❌ Missing PINECONE_INDEX_HOST"
assert PINECONE_NAMESPACE, "❌ Missing PINECONE_NAMESPACE"
assert MISTRALAI_API_KEY, "❌ Missing MISTRALAI_API_KEY"


def extract_region_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    region_div = soup.find("div", class_="region region-content")
    if region_div:
        return md(str(region_div))
    return md(html)


def extract_name_from_slug(slug: str) -> str:
    name_part = slug.replace("prof-", "").replace(".md", "")
    name_part = name_part.replace("-", " ")
    return name_part.title()


def extract_prof_metadata(text: str) -> dict:
    metadata = {}
    patterns = {
        'department': r"Department\s*\n\n(.*?)\n",
        'designation': r"Designation\s*\n\n(.*?)\n",
        'qualification': r"Qualification\s*\n\n(.*?)\n",
        'experience': r"Experience\s*\n\n(.*?)\n",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            metadata[key] = match.group(1).strip()
    return metadata


class MistralAPIException(Exception):
    pass


class RateLimitedEmbeddings:
    def __init__(self, base_embeddings, requests_per_minute: int = 60, batch_size: int = 32, max_retries: int = 5):
        self.base = base_embeddings
        self.requests_per_minute = max(1, requests_per_minute)
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self._min_interval_s = 60.0 / float(self.requests_per_minute)
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        self._last_call_ts = time.time()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            attempt = 0
            while True:
                try:
                    self._throttle()
                    res = self.base.embed_documents(chunk)
                    results.extend(res)
                    break
                except Exception as e:
                    message = str(e)
                    if "429" in message or "quota" in message.lower() or "rate" in message.lower():
                        attempt += 1
                        if attempt > self.max_retries:
                            raise
                        backoff_s = min(60, 2 ** attempt)
                        print(f"⚠️ Rate limited while embedding batch of {len(chunk)}. Retrying in {backoff_s}s (attempt {attempt}/{self.max_retries})...")
                        time.sleep(backoff_s)
                        continue
                    raise
        return results

    def embed_query(self, text: str) -> List[float]:
        attempt = 0
        while True:
            try:
                self._throttle()
                return self.base.embed_query(text)
            except Exception as e:
                message = str(e)
                if "429" in message or "quota" in message.lower() or "rate" in message.lower():
                    attempt += 1
                    if attempt > self.max_retries:
                        raise
                    backoff_s = min(60, 2 ** attempt)
                    print(f"⚠️ Rate limited while embedding query. Retrying in {backoff_s}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(backoff_s)
                    continue
                raise


def ingest_single_url(url: str) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; APSIT-Scraper/1.0)"
    }
    print(f"🕷️ Fetching: {url}")
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    markdown = extract_region_markdown(resp.text)
    if not markdown.strip():
        raise RuntimeError("Empty content extracted from page")

    slug = url.rstrip("/").split("/")[-1]
    professor_name = extract_name_from_slug(slug)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(markdown)

    base_metadata = extract_prof_metadata(markdown)
    base_metadata['professor_name'] = professor_name
    base_metadata['source'] = f"{slug}.md"

    docs: List[Document] = []
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata={**base_metadata, "doc_id": str(uuid.uuid4())}))

    print(f"✅ Prepared {len(docs)} chunks for '{professor_name}'")

    embeddings_base = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRALAI_API_KEY)
    embeddings = RateLimitedEmbeddings(embeddings_base, requests_per_minute=EMBED_RPM, batch_size=EMBED_BATCH_SIZE, max_retries=EMBED_MAX_RETRIES)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ = pc.Index(PINECONE_INDEX, host=PINECONE_INDEX_HOST)

    print("🚀 Uploading to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
        namespace=PINECONE_NAMESPACE,
    )
    print("🎉 Single-page ingestion complete.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/single_ingest.py <APSIT_FACULTY_URL>")
        sys.exit(1)
    ingest_single_url(sys.argv[1]) 