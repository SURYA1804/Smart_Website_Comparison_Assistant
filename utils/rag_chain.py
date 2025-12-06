from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from collections import Counter


def format_docs(docs):
    """Format documents so the model sees which website each chunk came from."""
    parts = []
    for d in docs:
        company = d.metadata.get("company_name", "Unknown company")
        url = d.metadata.get("source_url", "Unknown URL")
        parts.append(
            f"Company: {company}\nURL: {url}\nContent:\n{d.page_content}\n---"
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

Provide your answer with:
- Clear statement of which companies you are comparing (from context only)
- Key differences between them based on context
- Your recommendation with specific reasons from the context"""
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnablePassthrough() | retriever | format_docs,
        }
        | prompt
        | llm
    )
    return rag_chain


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


def query_websites(question, vectorstore, docs_per_company=10):
    """Query with per-company retrieval for balanced comparison"""
    
    # Get all unique companies dynamically
    companies = get_unique_companies(vectorstore)
    
    if not companies:
        print("Warning: No companies found. Using standard RAG chain.")
        # Fallback: use standard chain
        rag_chain = create_rag_chain(vectorstore)
        response = rag_chain.invoke(question)
        return response.content
    
    print(f"Found companies: {companies}")
    all_retrieved_docs = []
    
    # Retrieve separately for each company
    for company in companies:
        print(f"\n=== Retrieving from {company} ===")
        
        try:
            # Create retriever with metadata filter for this company
            company_retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": docs_per_company,
                    "filter": {"company_name": company}
                }
            )
            
            company_docs = company_retriever.invoke(question)
            print(f"  ✓ Retrieved {len(company_docs)} docs from {company}")
            
            # Debug: show first doc preview
            if company_docs:
                print(f"    Preview: {company_docs[0].page_content[:100]}...")
            
            all_retrieved_docs.extend(company_docs)
            
        except Exception as e:
            print(f"  ✗ Error with metadata filter: {e}")
            print(f"  → Trying without filter for {company}...")
            # Fallback: retrieve without filter
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
    
    # Create LLM
    llm = ChatGroq(
        model=os.getenv("Model_Name", "llama-3.3-70b-versatile"),
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    
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
7. Try to Include the Correct Numerical Term that mentioned in websites and highlight the main terms.
Provide your answer with:
- Clear statement of which companies you are comparing (from context only)
- Key differences between them based on context (at least 2-3 aspects)
- Your recommendation with specific reasons from the context
- Explanation of why you are recommending this option"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format context from all retrieved docs
    context = format_docs(all_retrieved_docs)
    
    # Get response
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    
    return response.content
