from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  

def create_vector_store(documents):
    """Create vector store from documents"""
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings (using free HuggingFace embeddings)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    return vectorstore
def delete_vector_store(vectorstore):
    """
    Safely delete a Chroma vector store and clean up resources
    
    Args:
        vectorstore: Chroma vectorstore instance to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    if vectorstore is None:
        return False
    
    try:
        # Get the collection name before deletion
        collection_name = vectorstore._collection.name
        
        # Delete the entire collection
        vectorstore._client.delete_collection(name=collection_name)
        
        print(f"✅ Successfully deleted collection: {collection_name}")
        return True
        
    except AttributeError as e:
        # Vector store doesn't have expected attributes
        print(f"⚠️ Vector store doesn't support deletion: {e}")
        return False
        
    except Exception as e:
        # Other errors
        print(f"❌ Error deleting vector store: {e}")
        return False


def reset_all_chroma_data(vectorstore):
    """
    ⚠️ DANGER: Reset entire Chroma database (deletes ALL collections)
    Use with extreme caution - this cannot be undone!
    
    Args:
        vectorstore: Any Chroma vectorstore instance (to access client)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if vectorstore is None:
        return False
    
    try:
        # Reset entire database
        vectorstore._client.reset()
        
        print("✅ All Chroma collections have been deleted")
        return True
        
    except Exception as e:
        print(f"❌ Error resetting Chroma database: {e}")
        return False