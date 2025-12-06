import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from utils.excel_loader import load_excel_file
from utils.web_scraper import scrape_websites
from utils.vector_store import create_vector_store
from utils.rag_chain import create_rag_chain, query_websites

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Website Comparison Assistant", layout="wide")

st.title("🔍 Website Comparison Assistant")
st.write("Upload an Excel file with websites and ask which is best for your needs")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'websites_loaded' not in st.session_state:
    st.session_state.websites_loaded = False

# File upload section
uploaded_file = st.file_uploader("Upload Excel file with website URLs", type=['xlsx', 'xls'])

if uploaded_file:
    # Load Excel file
    df = load_excel_file(uploaded_file)
    st.success(f"Loaded {len(df)} websites")
    st.dataframe(df)
    
    # Process websites button
    if st.button("Process Websites"):
        with st.spinner("Scraping websites..."):
            documents = scrape_websites(df)
            st.success(f"Scraped {len(documents)} pages")
        
        with st.spinner("Creating vector store..."):
            st.session_state.vector_store = create_vector_store(documents)
            st.session_state.websites_loaded = True
            st.success("Vector store created successfully!")

# Query section
if st.session_state.websites_loaded:
    st.divider()
    st.subheader("Ask Your Question")
    
    user_query = st.text_area(
        "Example: Which company provides the best insurance policy for families?",
        height=100
    )
    
    if st.button("Get Recommendation"):
        if user_query:
            with st.spinner("Analyzing websites..."):
                response = query_websites(
                    user_query, 
                    st.session_state.vector_store
                )
                
                st.divider()
                st.subheader("📊 Recommendation")
                st.write(response)
        else:
            st.warning("Please enter a question")
