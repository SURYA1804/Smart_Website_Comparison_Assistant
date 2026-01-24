

## 🌐 Smart Website Comparison Assistant 🚀


[![Smart Website Comparison Assistant](https://img.shields.io/badge/🚀_Live_Demo-Smart_Website_Comparison_Assistant-ff69b4?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartwebsitecomparisonassistant.streamlit.app/)  
[![Python](https://img.shields.io/badge/🐍-Python_3.9+-blue?style=for-the-badge&logo=python&logoColor=yellow)](https://python.org/)  
[![LangChain](https://img.shields.io/badge/🔗-LangChain_v1.1-green?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)  
[![Playwright](https://img.shields.io/badge/🎭-Playwright_v1.57-orange?style=for-the-badge&logo=playwright&logoColor=white)](https://playwright.dev/)  


> ⚡ **Smart Website Comparison Assistant** is an AI-powered research companion designed to simplify decision-making across multiple websites. Instead of manually browsing and comparing pages, users can upload a simple Excel file of URLs, and the system will scrape content in parallel using Playwright, process it into vector embeddings with ChromaDB + HuggingFace, and run a two-stage Retrieval-Augmented Generation (RAG) pipeline powered by Groq Llama 3.3. The result is a hallucination-free, context-driven comparison presented in clean markdown tables and structured summaries—helping users quickly identify which site best fits their requirements with clarity, speed, and precision.

---

## ✨ Features  

- 🔍 **Parallel Scraping at Scale** – Scrape 10+ pages per site simultaneously with Playwright + asyncio  
- 🧠 **Robust RAG Pipeline** – ChromaDB + HuggingFace embeddings + Groq Llama 3.3  
- ⚡ **Two-Stage LLM Processing** – Raw answer generation → Polished markdown formatting  
- 📊 **Live Progress Tracking** – Real-time scraping stats, logs, and performance metrics  
- 🎯 **Strict Context Enforcement** – Answers ONLY from scraped website content  
- 🗑️ **Automatic Cleanup** – ChromaDB collections reset between runs  
- 💎 **Beautiful Outputs** – Markdown tables, structured sections, and clear recommendations  

---

## 🛠 Tech Stack  

| Layer              | Tools & Libraries |
|--------------------|------------------|
| **Web Scraping**   | Playwright + asyncio |
| **Vector Database**| ChromaDB (persistent) |
| **Embeddings**     | sentence-transformers/all-MiniLM-L6-v2 |
| **LLM Inference**  | Groq (llama-3.3-70b-versatile) |
| **RAG Framework**  | LangChain + LCEL |
| **Content Parsing**| BeautifulSoup + lxml |
| **Async Handling** | aiohttp + asyncio |

---

## 🚀 Quick Start  

### 1. Clone & Install Dependencies  
```bash
git clone https://github.com/SURYA1804/Smart_Website_Comparison_Assistant.git
cd smart-website-comparison-assistant
pip install -r requirements.txt
playwright install chromium
```

### 2. Run the App  
```bash
streamlit run app.py
```

---

## 🎯 How It Works  

1. 📤 **Upload CSV** → `[company_name, website_url]`  
   - Example: `HDFC Bank → https://www.hdfcbank.com`  

2. 🌐 **Parallel Scraping** → Playwright + asyncio  
   - 10 concurrent pages per site  
   - 3 websites scraped in parallel batches  

3. 🧠 **Vector Store Creation** → ChromaDB  
   - RecursiveCharacterTextSplitter (1000/200)  
   - HuggingFace embeddings (MiniLM-L6-v2)  

4. 🤖 **RAG Query (Two-Stage LLM)**  
   - Stage 1: Generate raw answer (strict context)  
   - Stage 2: Format → Markdown tables + sections  

5. 💎 **Beautiful Output**  
   - ✅ Summary  
   - 📊 Comparison Table  
   - 🎯 Recommendation  

---

## 🌐 Live Demo  
👉 [Try it on Streamlit Cloud](https://smartwebsitecomparisonassistant.streamlit.app)  

---

## 📁 Project Structure  

```
smart-website-comparison-assistant
│
├── app.py              # Main Streamlit UI + orchestration
├── web_scraper.py      # Parallel Playwright scraper
├── vector_store.py     # ChromaDB + cleanup functions
├── rag_chain.py        # Two-stage RAG pipeline (LCEL)
├── utils.py            # Helper functions
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies
├── .env.example        # Environment template

```

---

## ⭐ Contribute & Support  

If you find this project helpful, please **star ⭐ the repo** and share it with others!  
Contributions, issues, and feature requests are welcome.  



