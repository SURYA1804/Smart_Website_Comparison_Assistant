from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from collections import Counter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter

from tenacity import retry, stop_after_attempt, wait_exponential
from collections import Counter
import streamlit as st
from langsmith import traceable


def format_docs(docs, max_chars_per_doc=800):  
    """Format documents with size limit"""
    parts = []
    for d in docs:
        company = d.metadata.get("company_name", "Unknown company")
        url = d.metadata.get("source_url", "Unknown URL")
        
        # Truncate content
        content = d.page_content[:max_chars_per_doc]
        if len(d.page_content) > max_chars_per_doc:
            content += "..."
        
        parts.append(
            f"Company: {company}\nURL: {url}\nContent:\n{content}\n---"
        )
    return "\n\n".join(parts)


def create_rag_chain(vectorstore):
    """Create RAG chain with standard retrieval (used for simple queries)"""
    llm = ChatGroq(
        model=os.getenv("Model_Name", "llama-3.3-70b-versatile"),
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    template = """You are a website comparison assistant. You MUST base your answer ONLY on the context provided below. Do NOT use any external knowledge or information about companies not mentioned in the context.

Context from websites:
{context}

Question: {question}

IMPORTANT RULES:
1. Only compare companies that appear in the context above
2. Only mention features, prices, and details explicitly stated in the context
3. If a company is not in the context, DO NOT mention it
4. Cite the company name and URL for each point you make
5. If you cannot answer based on the context, say "I don't have enough information from the provided websites"
6. If there is only information about a single website or company, then simply say "I don't have enough information about all provided websites to make a fair comparison"
7. If you dont have enough information information then just respond with I dont have enough information.Dont get hallucinated and providing the repeated words.
8. If context or page link not clear then dont mention like "page page "or "pagepage.html". Striclty avoid it
Provide your answer with:
- Clear statement of which companies you are comparing (from context only)
- Key differences between them based on context
- Your recommendation with specific reasons from the context"""
    formatting_template = """You are a professional content formatter. Your job is to take a raw comparison answer and format it beautifully with proper structure.

Raw Answer:
{raw_answer}

Format this response following these rules:

1. **Use clear markdown formatting:**
   - Use ## for main sections
   - Use ### for subsections
   - Use **bold** for company names
   - Use bullet points (-) for lists
   - Use tables for side-by-side comparisons

2. **Structure:**
   - Start with a brief summary (2-3 sentences)
   - Create clear sections: "Companies Compared", "Key Features", "Pricing", "Recommendations"
   - Use comparison tables when comparing 2-3 companies
   - End with a clear recommendation

3. **Clean up any issues:**
   - Remove any repeated words or phrases
   - Remove unclear references like "page page" or broken URLs
   - Ensure all company names are properly capitalized
   - Make sure all information flows logically

4. **Keep all factual information intact** - don't add or remove facts, just restructure

Provide the beautifully formatted response:"""
    
    answer_prompt = ChatPromptTemplate.from_template(template)
    formatting_prompt = ChatPromptTemplate.from_template(formatting_template)

    raw_answer_chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnablePassthrough() | retriever | format_docs,
        }
        | answer_prompt
        | llm
    )
    formatted_chain = (
        raw_answer_chain
        | (lambda x: {"raw_answer": x.content})
        | formatting_prompt
        | llm
    )

    return formatted_chain

def create_compression_retriever(base_retriever, llm):
    """Create compression retriever with LLMChainFilter"""
    if ContextualCompressionRetriever is None or LLMChainFilter is None:
        print("⚠️ Compression not available, using base retriever")
        return base_retriever
    
    try:
        # Create LLM filter to remove irrelevant content
        _filter = LLMChainFilter.from_llm(llm)
        
        # Wrap base retriever with compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_filter,
            base_retriever=base_retriever
        )
        
        print("✅ Using ContextualCompression with LLMChainFilter")
        return compression_retriever
    except Exception as e:
        print(f"⚠️ Failed to create compression retriever: {e}")
        return base_retriever

def get_unique_companies(vectorstore):
    """Extract unique company names from vector store dynamically"""
    try:
        # Access Chroma collection to get all metadata
        collection = vectorstore._collection
        results = collection.get(include=["metadatas"])
        
        companies = set()
        for metadata in results.get("metadatas", []):
            if "company_name" in metadata:
                companies.add(metadata["company_name"])
        
        return list(companies)
    except Exception as e:
        print(f"Could not extract companies dynamically: {e}")
        return []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def query_llm_with_retry(llm, messages):
    """Retry on API failures"""
    try:
        return llm.invoke(messages)
    except Exception as e:
        error_msg = str(e).lower()
        if "rate_limit" in error_msg or "429" in error_msg:
            print("⏳ API rate limit hit, retrying...")
            if 'st' in globals():  # Check if Streamlit is available
                st.warning("⏳ API rate limit hit, retrying...")
            raise  # Trigger retry
        elif "413" in error_msg or "too large" in error_msg:
            print("❌ Request too large")
            if 'st' in globals():
                st.error("❌ Request too large. Reduce document size.")
            raise ValueError("Request exceeds token limit")
        else:
            print(f"❌ Error: {str(e)}")
            if 'st' in globals():
                st.error(f"❌ Error: {str(e)}")
            raise


@traceable
def query_websites(question, vectorstore, docs_per_company=7, use_compression=True):
    """Query with per-company retrieval, compression, and retry logic"""
    
    # Get all unique companies dynamically
    companies = get_unique_companies(vectorstore)
    
    # Create LLM
    llm = ChatGroq(
        model=os.getenv("Model_Name", "llama-3.3-70b-versatile"),
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    
    if not companies:
        print("Warning: No companies found. Using standard RAG chain.")
        rag_chain = create_rag_chain(vectorstore)
        response = rag_chain.invoke(question)
        return response.content
    
    
    print(f"Found companies: {companies}")
    all_retrieved_docs = []
    
    # Retrieve separately for each company
    for company in companies:
        print(f"\n=== Retrieving from {company} ===")
        
        try:
            # Create base retriever with metadata filter
            base_retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": docs_per_company * 2 if use_compression else docs_per_company,  # Get more docs for compression
                    "filter": {"company_name": company}
                }
            )
            
            # Apply compression if enabled
            if use_compression:
                company_retriever = create_compression_retriever(base_retriever, llm)
            else:
                company_retriever = base_retriever
            
            # Retrieve documents
            company_docs = company_retriever.invoke(question)
            
            # If compression returned too many, limit
            if len(company_docs) > docs_per_company:
                company_docs = company_docs[:docs_per_company]
            
            print(f"  ✓ Retrieved {len(company_docs)} docs from {company}")
            
            # Debug: show first doc preview
            if company_docs:
                print(f"    Preview: {company_docs[0].page_content[:100]}...")
            
            all_retrieved_docs.extend(company_docs)
            
        except Exception as e:
            print(f"  ✗ Error with compression/filter: {e}")
            print(f"  → Trying fallback for {company}...")
            # Fallback: retrieve without compression/filter
            try:
                fallback_retriever = vectorstore.as_retriever(
                    search_kwargs={"k": docs_per_company}
                )
                docs = fallback_retriever.invoke(f"{question} {company}")
                # Filter manually
                company_docs = [
                    d for d in docs 
                    if d.metadata.get("company_name") == company
                ]
                if company_docs:
                    all_retrieved_docs.extend(company_docs[:docs_per_company])
                    print(f"  ✓ Retrieved {len(company_docs[:docs_per_company])} docs (manual filter)")
            except Exception as e2:
                print(f"  ✗ Fallback also failed: {e2}")
    
    # Check if we got any documents
    if not all_retrieved_docs:
        print("\n⚠️  No documents retrieved. Using standard RAG chain as fallback.")
        rag_chain = create_rag_chain(vectorstore)
        response = rag_chain.invoke(question)
        return response.content
    
    # Check distribution
    company_counts = Counter(
        doc.metadata.get("company_name", "Unknown") 
        for doc in all_retrieved_docs
    )
    
    print(f"\n=== DOCUMENT DISTRIBUTION ===")
    for company, count in company_counts.items():
        print(f"  {company}: {count} documents")
    print(f"=== TOTAL RETRIEVED: {len(all_retrieved_docs)} documents ===")
    print("=" * 50)
    
    # Check if we have info from multiple companies
    if len(company_counts) < 2:
        print("\n⚠️  WARNING: Retrieved documents from only one company!")
    
    # Create prompt template
    template = """You are a website comparison assistant. You MUST base your answer ONLY on the context provided below. Do NOT use any external knowledge or information about companies not mentioned in the context.

Context from websites:
{context}

Question: {question}

IMPORTANT RULES:
1. Only compare companies that appear in the context above
2. Only mention features, prices, and details explicitly stated in the context
3. If a company is not in the context, DO NOT mention it
4. Cite the company name for each point you make
5. If you cannot answer based on the context, say "I don't have enough information from the provided websites"
6. If there is only information about a single website or company, then say "I don't have enough information about all provided websites to make a fair comparison"
7. Include correct numerical terms mentioned in the websites and highlight the main terms

Provide your answer with:
- Clear statement of which companies you are comparing (from context only)
- Key differences between them based on context (at least 2-3 aspects)
- Your recommendation with specific reasons from the context
- Explanation of why you are recommending this option"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format context with size optimization
    context = format_docs_optimized(all_retrieved_docs, max_chars_per_doc=600)
    
    # Estimate tokens
    estimated_tokens = len(context) / 4
    print(f"Estimated context tokens: {int(estimated_tokens)}")
    
    # Safety check - truncate if too large
    if estimated_tokens > 4500:
        print("⚠️ Context too large, truncating...")
        context = context[:18000]
    
    # Get response with retry logic
    try:
        messages = prompt.format_messages(context=context, question=question)
        response = query_llm_with_retry(llm, messages)
        return response.content
    except ValueError as e:
        if "token limit" in str(e).lower():
            print("Retrying with reduced documents...")
            if 'st' in globals():
                st.info("Reducing document size and retrying...")
            return query_websites(question, vectorstore, docs_per_company=3, use_compression=False)
        else:
            raise
    except Exception as e:
        error_msg = f"Failed after multiple attempts: {str(e)}"
        print(error_msg)
        if 'st' in globals():
            st.error(error_msg)
        return "I apologize, but I encountered an error processing your request. Please try again later or with a simpler query."


def format_docs_optimized(docs, max_chars_per_doc=600):
    """Format documents with size limit"""
    parts = []
    for d in docs:
        company = d.metadata.get("company_name", "Unknown company")
        url = d.metadata.get("source_url", "Unknown URL")
        
        # Truncate content
        content = d.page_content[:max_chars_per_doc]
        if len(d.page_content) > max_chars_per_doc:
            content += "..."
        
        parts.append(f"Company: {company}\nURL: {url}\nContent:\n{content}\n---")
    
    return "\n\n".join(parts)
