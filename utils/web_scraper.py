from langchain_community.document_loaders import WebBaseLoader
import time

def scrape_websites(df):
    documents = []
    
    for idx, row in df.iterrows():
        url = row["website_url"]
        company = row["company_name"]
        print(f"\n[{idx+1}/{len(df)}] Loading {company} from {url}...")
        
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                header_template={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                requests_kwargs={"timeout": 20}
            )
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["company_name"] = company
                doc.metadata["source_url"] = url
            
            documents.extend(docs)
            print(f"  ✓ Loaded {len(docs)} doc(s)")
            print(f"  ✓ Content preview: {docs[0].page_content[:200]}...")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n=== TOTAL: {len(documents)} documents loaded ===")
    return documents
