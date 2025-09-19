import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www.apsit.edu.in"
visited = set()
to_visit = [BASE_URL]
faculty_links = []

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; XD/1.0; +https://www.apsit.edu.in/)"
}

while to_visit:
    url = to_visit.pop()
    if url in visited:
        continue

    try:
        response = requests.get(url, headers=headers, timeout=10)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            continue

        visited.add(url)
        print(f"🕵️‍♂️ Crawling: {url}")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Save faculty/profile links separately
        if "prof-" in url or "/faculty" in url:
            faculty_links.append(url)

        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href'].strip()

            # Skip non-navigational or junk links
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue

            full_url = urljoin(url, href)
            parsed_url = urlparse(full_url)

            # Skip CDN email protection, assets, duplicates, and offsite
            if (
                parsed_url.netloc == urlparse(BASE_URL).netloc and
                full_url not in visited and
                "cdn-cgi" not in full_url and
                not full_url.endswith(('.jpg', '.jpeg', '.png', '.pdf', '.docx', '.css', '.js')) and
                "#" not in parsed_url.path
            ):
                to_visit.append(full_url)

    except Exception as e:
        print(f"❌ Error at {url}: {e}")

# Save all visited links
with open("apsit_all_links.txt", "w", encoding="utf-8") as f:
    for link in sorted(visited):
        f.write(link + "\n")

# Save only faculty/profile pages
with open("apsit_faculty_links.txt", "w", encoding="utf-8") as f:
    for link in sorted(set(faculty_links)):
        f.write(link + "\n")

print(f"\n✅ Done. Found {len(visited)} total pages.")
print(f"📘 Found {len(faculty_links)} faculty/profile pages.")
