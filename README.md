# Smart Website Comparison Assistant 🚀

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartwebsitecomparisonassistant.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=yellow)](https://python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.1-green?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![Playwright](https://img.shields.io/badge/Playwright-v1.57-orange?style=for-the-badge&logo=playwright&logoColor=white)](https://playwright.dev/)

**AI-powered website comparison tool** that scrapes multiple websites in parallel, builds intelligent vector stores, and answers comparison questions using advanced RAG (Retrieval-Augmented Generation) with **two-stage LLM processing**.

## ✨ Features

- 🔍 **Lightning-Fast Parallel Scraping** - 10+ pages per site simultaneously using Playwright + asyncio
- 🧠 **Production-Ready RAG Pipeline** - ChromaDB + HuggingFace embeddings + Groq Llama 3.3
- ⚡ **Two-Stage LLM Processing** - Raw answer generation → Professional markdown formatting
- 📊 **Real-time Progress Tracking** - Live scraping stats, failure logs, and performance metrics
- 🎯 **Zero Hallucinations** - Answers ONLY from scraped website content (strict context enforcement)
- 🗑️ **Automatic Cleanup** - Proper ChromaDB collection deletion between scrapes
- 💎 **Beautiful Output** - Markdown tables, structured sections, and clear recommendations

## 🛠 Tech Stack

Web Scraping: Playwright 1.57.0 + asyncio
Vector Database: ChromaDB 1.3.5 (persistent)
Embeddings: sentence-transformers/all-MiniLM-L6-v2
LLM Inference: Groq (llama-3.3-70b-versatile)
RAG Framework: LangChain 1.1.0 + LCEL (LangChain Expression Language)
Content Processing: BeautifulSoup 4.14.2 + lxml
Async Processing: aiohttp 3.13.2 + asyncio



## 🚀 Quick Start (Local Development)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/SURYA1804/Smart_Website_Comparison_Assistant.git
cd smart-website-comparison-assistant
pip install -r requirements.txt
playwright install chromium
```

## 🎯 How It Works (Step-by-Step)
1. 📤 UPLOAD CSV → [company_name, website_url]
   └── HDFC Bank → https://www.hdfcbank.com
   
2. 🌐 PARALLEL SCRAPING → Playwright + asyncio
   ├── 10 concurrent pages per website
   ├── 3 websites in parallel batches
   └── Content cleaning (removes "Page Page Page")
   
3. 🧠 VECTOR STORE → ChromaDB
   ├── RecursiveCharacterTextSplitter (1000/200)
   ├── HuggingFace embeddings (all-MiniLM-L6-v2)
   └── Auto-delete previous collections
   
4. 🤖 RAG QUERY (Two-Stage LLM)
   ├── Stage 1: Generate raw answer (strict context)
   ├── Stage 2: Format → Markdown tables + sections
   └── Groq Llama 3.3 (llama-3.3-70b-versatile)
   
5. 💎 BEAUTIFUL OUTPUT
   └── ## Summary | ### Features | 📊 Comparison Table | ✅ Recommendation


## 🌐 Live Demo
[Streamlit Cloud
](https://smartwebsitecomparisonassistant.streamlit.app)

## 📁 Project Structure
smart-website-comparison-assistant<br>
│<br>
├── app.py                    # Main Streamlit UI + orchestration<br>
├── web_scraper.py            # Parallel Playwright scraper<br>
├── vector_store.py           # ChromaDB + cleanup functions<br>
├── rag_chain.py              # Two-stage RAG pipeline (LCEL)<br>
├── utils.py                  # Helper functions<br>
├── requirements.txt          # Python dependencies<br>
├── packages.txt              # system dependencies<br>
├── .env.example              # Environment template<br>

### ⭐ Star this repo if you found it helpful! ⭐