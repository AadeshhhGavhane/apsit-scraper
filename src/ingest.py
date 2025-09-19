import os
import glob
import uuid
import re  # Import the regular expressions library
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")
MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")

# Optional tuning for embedding rate limiting
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_RPM = int(os.getenv("EMBED_RPM", "60"))  # requests per minute
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))

# Pinecone batch size to stay under 2MB limit
PINECONE_BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE", "50"))  # Smaller batches for Pinecone

# Sanity checks
assert PINECONE_API_KEY, "❌ Missing PINECONE_API_KEY"
assert PINECONE_INDEX, "❌ Missing PINECONE_INDEX"
assert PINECONE_INDEX_HOST, "❌ Missing PINECONE_INDEX_HOST"
assert PINECONE_NAMESPACE, "❌ Missing PINECONE_NAMESPACE"
assert MISTRALAI_API_KEY, "❌ Missing MISTRALAI_API_KEY"


# --- START: NEW HELPER FUNCTIONS FOR SMART CHUNKING ---

def extract_name_from_filename(filename: str) -> str:
    """
    Parses a filename like 'prof-anupama-singh.md' into a proper name 'Anupama Singh'.
    """
    # Removes 'prof-' prefix and '.md' suffix, replaces hyphens with spaces
    name_part = filename.replace('prof-', '').replace('.md', '')
    return name_part.replace('-', ' ').title()

def extract_prof_metadata(text: str) -> dict:
    """
    Uses regular expressions to find and extract key details from the file content.
    """
    metadata = {}
    patterns = {
        'department': r"Department\s*\n\n(.*?)\n",
        'designation': r"Designation\s*\n\n(.*?)\n",
        'qualification': r"Qualification\s*\n\n(.*?)\n",
        'experience': r"Experience\s*\n\n(.*?)\n"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # .group(1) gets the captured text, .strip() removes whitespace
            metadata[key] = match.group(1).strip()
            
    return metadata

# --- END: NEW HELPER FUNCTIONS ---


# Setup text splitter (this part remains the same)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# --- START: MODIFIED DOCUMENT LOADING AND PROCESSING LOGIC ---

docs = []
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scraped_dir = os.path.join(root_dir, "scraped")
markdown_files = glob.glob(f"{scraped_dir}/*.md")

for path in markdown_files:
    filename = os.path.basename(path)
    print(f"📄 Processing file: {filename}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Extract the high-level, global metadata for this professor
    prof_metadata = extract_prof_metadata(text)
    prof_metadata['professor_name'] = extract_name_from_filename(filename)
    prof_metadata['source'] = filename  # Keep the source filename for reference
    
    # 2. Split the raw text into smaller chunks
    text_chunks = splitter.split_text(text)
    
    # 3. For each text chunk, create a Document with the rich, shared metadata
    for chunk_content in text_chunks:
        # Create a new Document object
        # The metadata is a combination of the professor's details and a unique ID
        doc = Document(
            page_content=chunk_content,
            metadata={
                **prof_metadata,  # Unpack the professor's metadata here
                "doc_id": str(uuid.uuid4())
            }
        )
        docs.append(doc)

print(f"✅ Loaded {len(markdown_files)} markdown files into {len(docs)} smart chunks")

# --- END: MODIFIED DOCUMENT LOADING ---


# Setup MistralAI embeddings with rate limiting and backoff
import time
from typing import List
try:
    from mistralai.exceptions import MistralAPIException
except Exception:  # Fallback if import path changes
    class MistralAPIException(Exception):
        pass

class RateLimitedEmbeddings:
    """Wrapper to throttle and retry embedding calls to avoid 429s."""

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
                except MistralAPIException as e:
                    message = str(e)
                    # Retry on common rate-limit indicators
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
            except MistralAPIException as e:
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

embeddings_base = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=MISTRALAI_API_KEY
)
embeddings = RateLimitedEmbeddings(
    embeddings_base,
    requests_per_minute=EMBED_RPM,
    batch_size=EMBED_BATCH_SIZE,
    max_retries=EMBED_MAX_RETRIES,
)

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX, host=PINECONE_INDEX_HOST)

# Process documents in batches to avoid Pinecone's 2MB limit
print(f"🚀 Starting batch upload to Pinecone (batch size: {PINECONE_BATCH_SIZE})...")

total_docs = len(docs)
uploaded_count = 0

for i in range(0, total_docs, PINECONE_BATCH_SIZE):
    batch_docs = docs[i:i + PINECONE_BATCH_SIZE]
    batch_num = (i // PINECONE_BATCH_SIZE) + 1
    total_batches = (total_docs + PINECONE_BATCH_SIZE - 1) // PINECONE_BATCH_SIZE
    
    print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)...")
    
    try:
        # Create vectorstore for this batch
        vectorstore = PineconeVectorStore.from_documents(
            documents=batch_docs,
            embedding=embeddings,
            index_name=PINECONE_INDEX,
            namespace=PINECONE_NAMESPACE
        )
        uploaded_count += len(batch_docs)
        print(f"✅ Batch {batch_num} uploaded successfully ({uploaded_count}/{total_docs} total)")
        
    except Exception as e:
        print(f"❌ Error uploading batch {batch_num}: {str(e)}")
        # Continue with next batch instead of failing completely
        continue

print(f"🎉 Data ingestion complete! Uploaded {uploaded_count}/{total_docs} documents to Pinecone.")
