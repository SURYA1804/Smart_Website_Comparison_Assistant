import pandas as pd

def load_excel_file(uploaded_file):
    """Load Excel file and extract website URLs"""
    df = pd.read_excel(uploaded_file)
    
    # Assume the Excel has columns: 'company_name', 'website_url'
    required_columns = ['company_name', 'website_url']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Excel must contain columns: {required_columns}")
    
    return df
