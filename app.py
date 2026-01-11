import streamlit as st
import pandas as pd
from io import BytesIO
import time
from dotenv import load_dotenv

from utils.excel_loader import load_excel_file
from utils.validate_file import validate_excel_input
from utils.web_scraper import scrape_websites
from utils.vector_store import create_vector_store, reset_all_chroma_data
from utils.rag_chain import query_websites

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Smart Website Comparison Assistant", 
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom CSS Styling
# --------------------------
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    
    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stat-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .stat-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Company cards */
    .company-card {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        transition: all 0.3s ease;
    }
    
    .company-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Dark theme answer container */
    .answer-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    .answer-container h1, .answer-container h2, .answer-container h3 {
        color: #8b9aff;
        margin-top: 1rem;
    }
    
    .answer-container h4, .answer-container h5, .answer-container h6 {
        color: #a5b4fc;
    }
    
    .answer-container p, .answer-container li, .answer-container span {
        color: #e0e0e0;
    }
    
    .answer-container strong {
        color: #ffffff;
    }
    
    .answer-container a {
        color: #8b9aff;
        text-decoration: underline;
    }
    
    .answer-container code {
        background: #0f1419;
        color: #8b9aff;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    
    .answer-container pre {
        background: #0f1419;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
    }
    
    .answer-container table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: #0f1419;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .answer-container th {
        background: #667eea;
        color: white;
        padding: 0.75rem;
        text-align: left;
    }
    
    .answer-container td {
        padding: 0.75rem;
        border-bottom: 1px solid #2d3748;
        color: #e0e0e0;
    }
    
    .answer-container tr:hover {
        background: #1a202c;
    }
    
    .answer-container blockquote {
        border-left: 3px solid #667eea;
        padding-left: 1rem;
        margin-left: 0;
        color: #cbd5e0;
        font-style: italic;
    }
    
    .answer-container ul, .answer-container ol {
        color: #e0e0e0;
    }
    
    .answer-container hr {
        border-color: #2d3748;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Load Environment
# --------------------------
load_dotenv()

# --------------------------
# Header Section
# --------------------------
st.markdown('<h1 class="main-title">🔍 Smart Website Comparison Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload websites and discover which is best for your needs</p>', unsafe_allow_html=True)

# --------------------------
# Session State Initialization
# --------------------------
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'websites_loaded' not in st.session_state:
    st.session_state.websites_loaded = False
if 'scraping_stats' not in st.session_state:
    st.session_state.scraping_stats = None
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'failures' not in st.session_state:
    st.session_state.failures = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'question_counter' not in st.session_state:
    st.session_state.question_counter = 0

# --------------------------
# Sidebar Stats
# --------------------------
with st.sidebar:
    st.header("📊 Session Statistics")
    
    if st.session_state.websites_loaded and st.session_state.scraping_stats:
        stats = st.session_state.scraping_stats
        
        st.metric("🌐 Total Pages", stats['total_pages_visited'])
        st.metric("✅ Successfully Scraped", stats['pages_scraped'], 
                 delta=f"{(stats['pages_scraped']/max(stats['total_pages_visited'],1)*100):.1f}%")
        st.metric("❌ Failed Pages", stats['pages_failed'])
        
        st.divider()
        
        if 'company_stats' in stats:
            st.subheader("📈 Per Company Details")
            for company, company_stats in stats['company_stats'].items():
                with st.expander(f"🏢 {company}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Pages", company_stats['pages_scraped'])
                    col2.metric("Time", f"{company_stats['time_elapsed']:.1f}s")
                    
                    if company_stats['time_elapsed'] > 0:
                        rate = company_stats['pages_scraped'] / company_stats['time_elapsed']
                        st.caption(f"Speed: {rate:.2f} pages/sec")
    else:
        st.info("📤 Upload and process websites to see statistics")
    
    st.divider()
    st.caption("💡 Powered by AI & LangChain")

# --------------------------
# Sample Excel Download Section
# --------------------------
with st.expander("📋 Need an Excel Template?", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("📝 Your Excel file must contain these two columns:")
        st.code("""
Column 1: company_name    (e.g., "Company A")
Column 2: website_url     (e.g., "https://example.com")
        """, language="text")
        
        st.warning("⚠️ Make sure URLs start with http:// or https://")
    
    with col2:
        sample_df = pd.DataFrame({
            'company_name': ['Company A', 'Company B', 'Company C'],
            'website_url': [
                'https://www.example1.com',
                'https://www.example2.com',
                'https://www.example3.com'
            ]
        })
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False, sheet_name='Websites')
        output.seek(0)
        
        st.download_button(
            "📥 Download Sample Template",
            data=output,
            file_name="website_comparison_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Download this template and fill in your company details"
        )

st.divider()

# --------------------------
# File Upload Section
# --------------------------
st.markdown("### 📤 Step 1: Upload Your Excel File")

uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx or .xls)",
    type=['xlsx', 'xls'],
    help="File must contain 'company_name' and 'website_url' columns"
)

if uploaded_file:
    try:
        df = load_excel_file(uploaded_file)
        
        # Validation
        errors = validate_excel_input(df)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.stop()
        
        st.success(f"✅ Successfully loaded {len(df)} website(s)!")
        
        # Display websites in attractive cards
        st.markdown("### 📋 Websites Ready for Scraping")
        
        # Create columns based on number of websites
        num_cols = min(3, len(df))
        cols = st.columns(num_cols)
        
        for idx, row in df.iterrows():
            with cols[idx % num_cols]:
                st.markdown(f"""
                    <div class="company-card">
                        <h4 style="margin:0; color: #667eea;">🏢 {row['company_name']}</h4>
                        <p style="margin:0.5rem 0 0 0; font-size:0.875rem; color: #6b7280; word-break: break-all;">
                            🔗 {row['website_url']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # --------------------------
        # Scraping Section
        # --------------------------
        st.markdown("### 🚀 Step 2: Start Web Scraping")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 Click below to start scraping. This may take a few minutes depending on website size.")
        with col2:
            scrape_button = st.button("🚀 Start Scraping", type="primary", use_container_width=True)
        
        if scrape_button:
            # Create containers for live updates
            st.session_state.vector_store = None
            st.session_state.websites_loaded = False
            st.session_state.scraping_stats = None
            st.session_state.documents = None
            st.session_state.failures = None
            st.session_state.selected_question = ""
            st.session_state.question_counter = 0
            status_container = st.container()
            metrics_container = st.container()
            progress_container = st.container()
            
            with status_container:
                st.markdown("### 🔄 Scraping in Progress...")
            
            # Initialize metrics display
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                metric_visited = col1.empty()
                metric_scraped = col2.empty()
                metric_failed = col3.empty()
                metric_current = col4.empty()
            
            # Progress bars
            with progress_container:
                overall_progress = st.progress(0)
                progress_text = st.empty()
            
            # Stats tracking
            live_stats = {
                "visited": 0,
                "scraped": 0,
                "failed": 0,
                "current_company": "",
                "total_expected": len(df) * 20
            }
            
            def update_progress(visited, total):
                """Update progress callback with live metrics"""
                live_stats["visited"] = visited
                live_stats["total_expected"] = total
                
                progress = min(visited / max(total, 1), 1.0)
                
                metric_visited.metric("📄 Pages Visited", visited)
                metric_scraped.metric("✅ Scraped", live_stats["scraped"])
                metric_failed.metric("❌ Failed", live_stats["failed"])
                metric_current.metric("🔍 Processing", "In Progress...")
                
                overall_progress.progress(progress)
                progress_text.markdown(f"**Progress:** {visited}/{total} pages • {progress*100:.1f}% complete")
            
            st.toast("🚀 Starting web scraping...", icon="🚀")
            
            with st.status("🌐 Scraping websites...", expanded=False) as status:
                start_time = time.time()
                
                try:
                    st.write("🔧 Initializing browser...")
                    
                    documents, failures, stats = scrape_websites(
                        df, 
                        progress_callback=update_progress,
                        batch_size=3
                    )
                    
                    # Store in session state
                    st.session_state.documents = documents
                    st.session_state.failures = failures
                    st.session_state.scraping_stats = stats
                    st.session_state.uploaded_df = df
                    
                    live_stats["scraped"] = stats['pages_scraped']
                    live_stats["failed"] = stats['pages_failed']
                    live_stats["visited"] = stats['total_pages_visited']
                    
                    metric_visited.metric("📄 Pages Visited", stats['total_pages_visited'])
                    metric_scraped.metric("✅ Scraped", stats['pages_scraped'])
                    metric_failed.metric("❌ Failed", stats['pages_failed'])
                    metric_current.metric("🔍 Status", "Complete! ✅")
                    
                    overall_progress.progress(1.0)
                    progress_text.markdown(
                        f"**✅ Complete:** {stats['pages_scraped']} pages scraped, "
                        f"{stats['pages_failed']} failed out of {stats['total_pages_visited']} visited"
                    )
                    
                    elapsed = time.time() - start_time
                    
                    st.write(f"✅ Scraping completed in {elapsed:.1f} seconds!")
                    st.write(f"📊 Successfully scraped {stats['pages_scraped']} pages")
                    
                    status.update(label="✅ Scraping Complete!", state="complete")
                    
                except Exception as e:
                    st.error(f"❌ Error during scraping: {str(e)}")
                    status.update(label="❌ Scraping Failed", state="error")
                    st.stop()
            
            st.toast("✅ Scraping completed successfully!", icon="✅")
            
            # Create vector store
            st.divider()
            
            with st.spinner("🔮 Creating AI-powered search index..."):
                st.toast("🔮 Building vector database...", icon="🔮")
                
                st.session_state.vector_store = create_vector_store(documents)
                st.session_state.websites_loaded = True
                
                st.success("✅ Vector store created successfully!")
                st.toast("✅ Ready to answer your questions!", icon="✨")
            
            st.balloons()
            st.rerun()

    except ValueError as e:
        st.error(f"❌ Invalid Excel format: {str(e)}")
        st.stop()

# --------------------------
# Scraping Results Section (OUTSIDE upload block)
# --------------------------
if st.session_state.scraping_stats is not None:
    st.divider()
    st.markdown("### 📊 Scraping Results")
    
    stats = st.session_state.scraping_stats
    
    # Clear data button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            response = reset_all_chroma_data(st.session_state.vector_store)
            with st.spinner("🗑️ Cleaning up previous vector store..."):
                if response:
                    st.success("✅ Previous vector store deleted")
                else:
                    st.info("ℹ️ Cleared vector store reference")
            st.session_state.vector_store = None
            st.session_state.websites_loaded = False
            st.session_state.scraping_stats = None
            st.session_state.documents = None
            st.session_state.failures = None
            st.session_state.uploaded_df = None
            st.session_state.selected_question = ""

            st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <h3>{stats['total_pages_visited']}</h3>
                <p>📄 Total Pages Visited</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                <h3>{stats['pages_scraped']}</h3>
                <p>✅ Successfully Scraped</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fail_color = "#ef4444" if stats['pages_failed'] > 0 else "#10b981"
        st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, {fail_color} 0%, {fail_color}dd 100%);">
                <h3>{stats['pages_failed']}</h3>
                <p>❌ Failed Pages</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Additional stats
    if stats.get('pages_per_domain'):
        with st.expander("🌐 Detailed Domain Statistics"):
            domain_df = pd.DataFrame([
                {
                    "Domain": domain,
                    "Pages Scraped": count,
                    "Percentage": f"{(count/stats['pages_scraped']*100):.1f}%"
                }
                for domain, count in stats['pages_per_domain'].items()
            ]).sort_values('Pages Scraped', ascending=False)
            
            st.dataframe(domain_df, use_container_width=True, hide_index=True)
    
    # Show failures if any
    if st.session_state.failures:
        failures = st.session_state.failures
        with st.expander(f"⚠️ Failed URLs ({len(failures)} pages)", expanded=False):
            st.warning("These pages could not be scraped. Common reasons: timeouts, blocking, or network errors.")
            
            failures_df = pd.DataFrame(failures)
            if 'company' in failures_df.columns:
                failures_df = failures_df[['company', 'url', 'status', 'error']]
            
            st.dataframe(failures_df, use_container_width=True, hide_index=True)

# --------------------------
# Query Section
# --------------------------
if st.session_state.websites_loaded:
    st.divider()
    st.markdown("### 💬 Step 3: Ask Your Questions")
    
    # Example questions
    with st.expander("💡 Click here for example questions"):
        st.markdown("**Try asking questions like:**")
        
        example_questions = [
            "Which company provides the best insurance policy for families?",
            "Compare pricing plans across all companies",
            "What are the key features offered by each company?",
            "Which company has the best customer support?",
            "Show me the main differences between these companies",
            "What are the unique selling points of each company?"
        ]
        
        cols = st.columns(2)
        for idx, q in enumerate(example_questions):
            with cols[idx % 2]:
                if st.button(f"📝 {q}", key=f"example_{idx}", use_container_width=True):
                    st.session_state.selected_question = q
                    st.session_state.question_counter += 1
                    st.rerun()
    
    st.divider()
    
    # Query input with dynamic key to force update
    user_query = st.text_area(
        "✍️ Enter your question:",
        value=st.session_state.selected_question,
        height=100,
        placeholder="e.g., Which company offers the best value for money for small businesses?",
        help="Ask any question about the websites you've scraped. Be specific for better results!",
        key=f"query_input_{st.session_state.question_counter}"
    )
    
    col1, col2, col3 = st.columns([2, 1, 3])
    
    with col1:
        ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True, disabled=not user_query)
    
    with col2:
        if st.button("🔄 Clear Question", use_container_width=True):
            st.session_state.selected_question = ""
            st.session_state.question_counter += 1
            st.rerun()
    
    if ask_button and user_query:
        with st.spinner("🤔 Analyzing websites and generating answer..."):
            st.toast("🤖 AI is processing your question...", icon="🤖")
            
            try:
                response = query_websites(user_query, st.session_state.vector_store)
                
                st.divider()
                st.markdown("### 💡 Answer")
                
                st.markdown(f'<div class="answer-container">{response}</div>', unsafe_allow_html=True)
                
                st.toast("✅ Answer generated successfully!", icon="✨")
                
                # Show source documents
                with st.expander("📚 View Source Documents"):
                    st.caption("These are the relevant sections from the websites used to generate the answer:")
                    
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    docs = retriever.invoke(user_query)
                    
                    for idx, doc in enumerate(docs, 1):
                        company = doc.metadata.get('company_name', 'Unknown')
                        url = doc.metadata.get('source_url', 'N/A')
                        word_count = doc.metadata.get('word_count', 'N/A')
                        
                        st.markdown(f"**{idx}. {company}**")
                        st.caption(f"🔗 {url}")
                        if word_count != 'N/A':
                            st.caption(f"📊 Word count: {word_count}")
                        
                        with st.container():
                            st.text(doc.page_content[:300] + "...")
                        
                        if idx < len(docs):
                            st.divider()
                
                # Feedback section
                st.divider()
                st.markdown("#### 📋 Was this answer helpful?")
                
                col1, col2, col3 = st.columns([1, 1, 3])
                
                with col1:
                    if st.button("👍 Yes, Helpful", use_container_width=True):
                        st.success("Thank you for your feedback! 😊")
                
                with col2:
                    if st.button("👎 Not Helpful", use_container_width=True):
                        st.info("We'll work on improving our answers! 🔧")
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.info("💡 Try rephrasing your question or check if the vector store was created successfully.")
    
    elif ask_button and not user_query:
        st.warning("⚠️ Please enter a question before clicking 'Get Answer'")

else:
    # Call to action when no data is loaded
    st.info("👆 **Getting Started:** Upload an Excel file with company websites and click 'Start Scraping' to begin!")
    
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        ### 🎯 How to Use This Tool
        
        1. **Upload Excel File** 📤
           - Prepare an Excel file with two columns: `company_name` and `website_url`
           - Upload it using the file uploader above
        
        2. **Start Scraping** 🚀
           - Click the "Start Scraping" button
           - Our AI will crawl up to 20 pages per website
           - This process runs in parallel for faster results
        
        3. **Ask Questions** 💬
           - Once scraping is complete, ask any comparison questions
           - Get AI-powered insights and recommendations
           - View source documents for transparency
        
        ### ✨ Features
        
        - **Parallel Scraping**: Process multiple websites simultaneously
        - **AI-Powered Analysis**: Get intelligent comparisons and recommendations
        - **Source Transparency**: See exactly which web pages were used
        - **Real-time Progress**: Monitor scraping progress live
        """)

# --------------------------
# Footer
# --------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>🔍 <strong>Smart Website Comparison Assistant</strong></p>
        <p style="font-size: 0.875rem;">Made with ❤️ for smart decision-making</p>
    </div>
""", unsafe_allow_html=True)
