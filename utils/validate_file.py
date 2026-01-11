import validators
from urllib.parse import urlparse

def validate_excel_input(df):
    """Validate uploaded data"""
    errors = []
    
    # Check columns
    if not all(col in df.columns for col in ['company_name', 'website_url']):
        errors.append("Missing required columns")
    
    # Validate URLs
    for idx, row in df.iterrows():
        url = row.get('website_url', '')
        
        # Check valid URL
        if not validators.url(url):
            errors.append(f"Invalid URL: {url}")
        
        # Check domain not localhost/internal
        domain = urlparse(url).netloc
        if domain in ['localhost', '127.0.0.1'] or domain.startswith('192.168'):
            errors.append(f"Local URLs not allowed: {url}")
        
        # Limit to HTTPS only
        if not url.startswith('https://'):
            errors.append(f"Only HTTPS allowed: {url}")
    
    # Limit number of sites
    if len(df) > 10:
        errors.append("Maximum 10 websites allowed")
    
    return errors


