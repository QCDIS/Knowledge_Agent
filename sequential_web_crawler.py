import time
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import HTTPError

ALLOWED_SCHEMES = {"http", "https"}

def make_session():
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return s

def get_all_links(url, same_domain_only=True):
    """
    Returns a sorted list of absolute, deduped, fragment-free URLs
    (optionally restricted to the same domain).
    """
    session = make_session()
    # small pause can help with WAF/rate-limits
    time.sleep(0.8)

    resp = session.get(url, timeout=25)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    origin_host = urlparse(url).netloc
    found = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        print("Working ...")

        # skip non-web links and empty anchors
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        # build absolute URL
        abs_url = urljoin(url, href)

        # drop URL fragments (#section)
        abs_url, _frag = urldefrag(abs_url)

        parsed = urlparse(abs_url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            continue

        if same_domain_only and parsed.netloc != origin_host:
            continue

        found.add(abs_url)

    return sorted(found)

# https://gcos.wmo.int/ar/node/24867
if __name__ == "__main__":
    # start_url = "https://gcos.wmo.int/site/global-climate-observing-system-gcos/essential-climate-variables"
    # start_url = " https://goosocean.org/what-we-do/framework/essential-ocean-variables/"
    # start_url = " https://vocab.nerc.ac.uk/collection/EXV/current/"
    
    with open("RI_Data/RI/urls.txt", "r") as file:
            RI_urls = [line.strip() for line in file if line.strip()]
    print("RI URLs ", RI_urls)
    all_urls = []
    # FOR LOOP
    
    try:
        for url in RI_urls:
            all_urls += get_all_links(url, same_domain_only=True)

        out_path = "explored_urls.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for link in all_urls:
                f.write(link + "\n")
        print(f"Saved {len(all_urls)} links to {out_path}")

    except HTTPError as e:
        print(f"HTTP error: {e}")  # includes status code and URL
    except requests.RequestException as e:
        # Any other requests-related error (connection, timeout, etc.)
        print(f"Request failed: {e}")

