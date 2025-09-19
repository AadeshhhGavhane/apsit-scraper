import os
import sys
import psutil
import asyncio
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Base paths from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRAPED_DIR = os.path.join(BASE_DIR, "scraped")
URL_FILE = os.path.join(BASE_DIR, "apsit_faculty_links.txt")

os.makedirs(SCRAPED_DIR, exist_ok=True)

# Load URLs from txt file
def get_urls(filepath=URL_FILE):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Create clean filenames like prof-xyz.md
def get_filename_from_url(url):
    slug = url.rstrip("/").split("/")[-1]
    return f"{slug}.md"


# Extract region-content as markdown
def extract_region_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    region_div = soup.find("div", class_="region region-content")
    if region_div:
        return md(str(region_div))
    return ""

# Log memory stats
def memory_logger(process, peak_memory, prefix=""):
    current = process.memory_info().rss
    if current > peak_memory[0]:
        peak_memory[0] = current
    print(f"{prefix}Memory: {current // (1024 * 1024)} MB, Peak: {peak_memory[0] // (1024 * 1024)} MB")

# Async scrape in batches
async def crawl_parallel(urls, max_concurrent=5):
    print(f"🕷️ Crawling {len(urls)} URLs (max {max_concurrent} parallel)...\n")

    peak_memory = [0]
    process = psutil.Process(os.getpid())

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    success, fail = 0, 0

    try:
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i:i + max_concurrent]
            tasks = []

            memory_logger(process, peak_memory, f"Before batch {i//max_concurrent + 1}: ")

            for j, url in enumerate(batch):
                session_id = f"session_{i + j}"

                async def scrape(url=url, session_id=session_id):
                    nonlocal success, fail
                    try:
                        result = await crawler.arun(url=url, config=crawl_config, session_id=session_id)
                        md_content = extract_region_content(result.html) or result.markdown
                        filename = get_filename_from_url(url)
                        filepath = os.path.join(SCRAPED_DIR, filename)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(md_content)
                        print(f"✅ Saved: {filename}")
                        success += 1
                    except Exception as e:
                        print(f"❌ Failed: {url} — {e}")
                        fail += 1

                tasks.append(scrape())

            await asyncio.gather(*tasks)
            memory_logger(process, peak_memory, f"After batch {i//max_concurrent + 1}: ")

    finally:
        await crawler.close()
        memory_logger(process, peak_memory, "Final: ")
        print(f"\n🧾 Summary:\n  ✅ Success: {success}\n  ❌ Failures: {fail}")
        print(f"📈 Peak RAM: {peak_memory[0] // (1024 * 1024)} MB")

# Run
if __name__ == "__main__":
    urls = get_urls()
    if not urls:
        print("No URLs found.")
    else:
        asyncio.run(crawl_parallel(urls, max_concurrent=10))
