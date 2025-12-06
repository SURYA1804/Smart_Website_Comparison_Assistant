from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session_with_retries():
    """Create requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def bs4_extractor(html: str) -> str:
    """Extract text content from HTML using BeautifulSoup"""
    soup = BeautifulSoup(html, "lxml")
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator="\n", strip=True)


def scrape_websites(df, max_depth=1, max_pages_per_site=20):
    """Scrape with better error handling"""
    documents = []
    session = create_session_with_retries()
    
    for idx, row in df.iterrows():
        url = row["website_url"]
        company = row["company_name"]
        print(f"\n[{idx+1}/{len(df)}] Testing connection to {company}...")
        
        # First test with simple request
        try:
            test_response = session.get(url, timeout=15)
            print(f"  ✓ Connection successful (status: {test_response.status_code})")
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            print(f"  → Skipping {company}")
            continue
        
        # Now try scraping
        print(f"  → Starting crawl of {company}...")
        start_time = time.time()
        
        try:
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=max_depth,
                extractor=bs4_extractor,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                prevent_outside=True,
            )
            
            docs = loader.load()
            
            if len(docs) > max_pages_per_site:
                docs = docs[:max_pages_per_site]
            
            for doc in docs:
                doc.metadata["company_name"] = company
                doc.metadata["source_url"] = url
            
            documents.extend(docs)
            
            elapsed = time.time() - start_time
            print(f"  ✓ Loaded {len(docs)} page(s) in {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Scraping error after {elapsed:.1f}s: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(documents)} pages loaded from {len(df)} websites")
    print(f"{'='*60}")
    
    return documents
