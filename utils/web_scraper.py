from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from collections import defaultdict
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import time
from datetime import datetime
from enum import Enum
import subprocess
import os

def install_playwright_browsers():
    """Install Playwright browsers automatically on Streamlit Cloud"""
    try:
        # Install chromium browser
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True)
        print("✅ Playwright chromium installed!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Playwright install error: {e}")

# Only install once
if "PLAYWRIGHT_INSTALLED" not in os.environ:
    install_playwright_browsers()
    os.environ["PLAYWRIGHT_INSTALLED"] = "true"

# ✅ Windows fix for Playwright + asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if "PLAYWRIGHT_INSTALLED" not in os.environ:
    install_playwright_browsers()
    os.environ["PLAYWRIGHT_INSTALLED"] = "true"
# --------------------------
# Console colors for debugging
# --------------------------
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class PageStatus(Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"
    BLOCKED = "BLOCKED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

# --------------------------
# Enhanced logging
# --------------------------
def log_info(message):
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {message}")

def log_success(message):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")

def log_warning(message):
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")

def log_error(message):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")

def log_debug(message):
    print(f"{Colors.OKCYAN}[DEBUG]{Colors.ENDC} {message}")

# --------------------------
# Extract internal links (optimized)
# --------------------------
def extract_internal_links(base_url, html):
    """Extract internal links from HTML content"""
    try:
        soup = BeautifulSoup(html, "lxml")
        base_domain = urlparse(base_url).netloc
        links = set()
        
        # Pre-compile exclusions
        excluded_extensions = {'.pdf', '.jpg', '.png', '.zip', '.doc', '.docx', '.gif', '.mp4'}
        
        for a in soup.find_all("a", href=True, limit=100):  # Limit links to prevent explosion
            try:
                href = urljoin(base_url, a["href"])
                parsed = urlparse(href)
                
                if (parsed.scheme in ["http", "https"] and 
                    parsed.netloc == base_domain and
                    not any(parsed.path.endswith(ext) for ext in excluded_extensions)):
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                    links.add(clean_url)
            except:
                continue
        
        return links
    except Exception as e:
        log_debug(f"Error extracting links: {e}")
        return set()

# --------------------------
# Optimized page loader with aggressive parallelism
# --------------------------
async def load_page_fast(context, url, company, retry_count=2):
    """Load a single page with optimized retry logic"""
    
    for attempt in range(retry_count):
        page = await context.new_page()
        
        try:
            log_info(f"[{company}] Attempt {attempt + 1} - Loading: {url}")
            
            # Navigate with shorter timeout
            response = await page.goto(
                url,
                wait_until="domcontentloaded",  # Faster than networkidle
                timeout=30000  # 30 seconds
            )
            
            # Quick status check
            if response and response.status >= 400:
                log_error(f"HTTP {response.status}")
                await page.close()
                if attempt < retry_count - 1:
                    await asyncio.sleep(1)
                    continue
                return None, PageStatus.NETWORK_ERROR, f"HTTP {response.status}"
            
            # Minimal wait for dynamic content
            await asyncio.sleep(0.5)
            
            # Get content
            html = await page.content()
            
            # Extract text quickly
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'meta', 'link', 'noscript', 'iframe']):
                tag.decompose()
            
            text = " ".join(soup.get_text(separator=" ").split())
            word_count = len(text.split())
            
            log_debug(f"Extracted {word_count} words")
            
            # Quick validation
            if word_count < 50:
                log_warning(f"Content too short ({word_count} words)")
                await page.close()
                if attempt < retry_count - 1:
                    await asyncio.sleep(1)
                    continue
                return None, PageStatus.CONTENT_TOO_SHORT, f"Only {word_count} words"
            
            # Check blocking
            text_lower = text.lower()
            if any(kw in text_lower for kw in ["access denied", "captcha", "cloudflare", "blocked"]):
                log_warning(f"Blocking detected")
                await page.close()
                return None, PageStatus.BLOCKED, "Blocking keywords"
            
            log_success(f"Loaded: {url} ({word_count} words)")
            await page.close()
            return (html, text), PageStatus.SUCCESS, None
            
        except PlaywrightTimeout:
            log_error(f"Timeout")
            await page.close()
            if attempt < retry_count - 1:
                await asyncio.sleep(1)
                continue
            return None, PageStatus.TIMEOUT, "Timeout"
            
        except Exception as e:
            error_msg = str(e)[:100]
            log_error(f"Error: {error_msg}")
            await page.close()
            if attempt < retry_count - 1:
                await asyncio.sleep(1)
                continue
            return None, PageStatus.UNKNOWN_ERROR, error_msg
    
    return None, PageStatus.UNKNOWN_ERROR, "Max retries"

# --------------------------
# Parallel crawl for single company
# --------------------------
async def crawl_site_parallel(browser, base_url, company, max_pages=20, concurrency=10):
    """Crawl a website with massive parallelism"""
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}Starting parallel crawl: {company}{Colors.ENDC}")
    print(f"{Colors.HEADER}Concurrency: {concurrency}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    # Create isolated browser context for this company
    context = await browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    )
    
    visited = set()
    to_visit = {base_url}
    documents = []
    failures = []
    domain_counts = defaultdict(int)
    status_counts = defaultdict(int)
    semaphore = asyncio.Semaphore(concurrency)
    
    start_time = time.time()
    
    async def load_page_safe(url):
        async with semaphore:
            return await load_page_fast(context, url, company)
    
    # Process in waves for better control
    while to_visit and len(visited) < max_pages:
        # Take batch of URLs to process
        batch_size = min(concurrency * 2, max_pages - len(visited))
        batch = list(to_visit)[:batch_size]
        
        log_info(f"Processing batch of {len(batch)} URLs...")
        
        # Process batch in parallel
        tasks = []
        for url in batch:
            if url not in visited and len(visited) < max_pages:
                visited.add(url)
                to_visit.discard(url)
                tasks.append((url, load_page_safe(url)))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for (url, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                log_error(f"Exception for {url}: {result}")
                failures.append({"url": url, "status": "EXCEPTION", "error": str(result), "company": company})
                continue
            
            content, status, error = result
            status_counts[status] += 1
            
            if content is None:
                failures.append({"url": url, "status": status.value, "error": error, "company": company})
                continue
            
            html, text = content
            domain = urlparse(url).netloc
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "company_name": company,
                    "source_url": url,
                    "domain": domain,
                    "word_count": len(text.split()),
                    "scraped_at": datetime.now().isoformat(),
                }
            )
            documents.append(doc)
            domain_counts[domain] += 1
            
            # Extract links for next batch
            new_links = extract_internal_links(base_url, html) - visited
            to_visit.update(new_links)
    
    await context.close()
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}Summary: {company}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    log_success(f"Pages: {len(documents)}, Failed: {len(failures)}, Time: {elapsed:.2f}s")
    
    stats = {
        "total_pages_visited": len(visited),
        "pages_scraped": len(documents),
        "pages_failed": len(failures),
        "pages_per_domain": dict(domain_counts),
        "status_breakdown": {k.value: v for k, v in status_counts.items()},
        "time_elapsed": elapsed
    }
    
    return documents, failures, stats

# --------------------------
# Multi-website PARALLEL scraper
# --------------------------
async def scrape_websites_async(df, progress_callback=None, batch_size=3):
    """Scrape ALL websites in parallel with batching"""
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}PARALLEL MULTI-WEBSITE SCRAPER{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}Websites: {len(df)}, Batch size: {batch_size}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    all_docs, all_failures = [], []
    total_stats = {
        "total_pages_visited": 0,
        "pages_scraped": 0,
        "pages_failed": 0,
        "pages_per_domain": defaultdict(int),
        "company_stats": {}
    }
    
    overall_start = time.time()
    
    async with async_playwright() as p:
        log_info("Launching browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security'
            ]
        )
        log_success("Browser launched")
        
        # Process websites in batches to prevent resource exhaustion
        for batch_idx in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_idx:batch_idx + batch_size]
            
            log_info(f"Processing batch {batch_idx//batch_size + 1} ({len(batch_df)} websites)...")
            
            # Create tasks for parallel execution
            tasks = []
            for idx, row in batch_df.iterrows():
                company = row.company_name
                url = row.website_url.rstrip("/")
                
                log_info(f"Queuing: {company}")
                task = crawl_site_parallel(browser, url, company, max_pages=20, concurrency=10)
                tasks.append((company, task))
            
            # Execute all companies in this batch IN PARALLEL
            log_success(f"Running {len(tasks)} websites in parallel...")
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            for (company, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    log_error(f"Failed to scrape {company}: {result}")
                    continue
                
                docs, fails, stats = result
                
                all_docs.extend(docs)
                all_failures.extend(fails)
                
                total_stats["total_pages_visited"] += stats["total_pages_visited"]
                total_stats["pages_scraped"] += stats["pages_scraped"]
                total_stats["pages_failed"] += stats["pages_failed"]
                total_stats["company_stats"][company] = stats
                
                for domain, count in stats["pages_per_domain"].items():
                    total_stats["pages_per_domain"][domain] += count
                
                if progress_callback:
                    progress_callback(total_stats["total_pages_visited"], len(df) * 20)
        
        log_info("Closing browser...")
        await browser.close()
        log_success("Browser closed")
    
    overall_elapsed = time.time() - overall_start
    
    # Final summary
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}FINAL SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    log_success(f"Websites: {len(df)}, Total time: {overall_elapsed:.2f}s")
    log_success(f"Pages visited: {total_stats['total_pages_visited']}")
    log_success(f"Pages scraped: {total_stats['pages_scraped']}")
    log_success(f"Average per website: {overall_elapsed/len(df):.2f}s")
    log_info(f"Speed: {total_stats['pages_scraped']/overall_elapsed:.1f} pages/second")
    
    print(f"\n{Colors.BOLD}Per company breakdown:{Colors.ENDC}")
    for company, stats in total_stats["company_stats"].items():
        rate = stats['pages_scraped'] / stats['time_elapsed'] if stats['time_elapsed'] > 0 else 0
        print(f"  - {company}: {stats['pages_scraped']} pages in {stats['time_elapsed']:.1f}s ({rate:.1f} pages/s)")
    
    return all_docs, all_failures, total_stats

# --------------------------
# Chunk documents (optimized)
# --------------------------
def chunk_documents(documents):
    """Split documents into chunks efficiently"""
    log_info(f"Chunking {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(doc.page_content):
            chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    
    log_success(f"Created {len(chunks)} chunks")
    return chunks

# --------------------------
# Main entry point
# --------------------------
def scrape_websites(df, progress_callback=None, batch_size=3):
    """
    Main entry point for parallel scraping
    
    Args:
        df: DataFrame with company_name and website_url
        progress_callback: Callback for progress updates
        batch_size: Number of websites to scrape in parallel (3-5 recommended)
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    raw_docs, failures, stats = loop.run_until_complete(
        scrape_websites_async(df, progress_callback, batch_size)
    )
    
    return chunk_documents(raw_docs), failures, stats
