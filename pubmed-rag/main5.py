import os
from Bio import Entrez
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
Entrez.email = os.environ.get("EMAIL_ID")
NCBI_API_KEY = os.environ.get("API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")


def fetch_pubmed_articles(query, max_results=1):
    print(f"🔎 Searching PubMed for: {query}...")
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="pub_date")
        record = Entrez.read(handle)
        id_list = record["IdList"]
        if not id_list: return []

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        articles = Entrez.read(handle)
        
        docs = []
        for article in articles['PubmedArticle']:
            pmid = article['MedlineCitation']['PMID']
            title = article['MedlineCitation']['Article']['ArticleTitle']
            abstract_parts = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
            abstract_text = " ".join([str(p) for p in abstract_parts])
            
            docs.append(Document(
                page_content=f"Title: {title}\nAbstract: {abstract_text}",
                metadata={"source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "title": title}
            ))
            print(f"✅ Found: {title[:70]}...")
        return docs
    except Exception as e:
        print(f"❌ PubMed Error: {e}")
        return []

def create_vector_db(docs):
    print("🧠 Generating Embeddings (Manual Batching)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    if not splits: return None

    try:
        # Initialize Chroma with a persistent directory to avoid memory issues
        vectorstore = Chroma(
            collection_name="pubmed_research",
            embedding_function=embeddings,
            persist_directory="./chroma_db" 
        )

        # Process 2 chunks at a time to stay under free-tier limits
        batch_size = 2
        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            vectorstore.add_documents(batch)
            time.sleep(1) # Rate limit protection
            
        return vectorstore
    except Exception as e:
        print(f"❌ Embedding Error: {e}")
        return None

def ask_medical_bot(question, vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.invoke(question)
    
    context_text = "\n\n".join([
        f"SOURCE: {d.metadata['title']}\nCONTENT: {d.page_content}" 
        for d in context_docs
    ])

    template = """
    You are a professional Medical Research Assistant. 
    Using the context below, provide a brief, high-level summary.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR RESPONSE (Keep it brief and professional):
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    # We invoke and then explicitly clean the output
    response = chain.invoke({"context": context_text, "question": question})
    
    # CLEANING LOGIC: Extract only the text part
    if isinstance(response.content, list):
        return response.content[0].get('text', 'No text found.')
    return response.content

if __name__ == "__main__":
    topic = input("💬 Enter Medical Topic: ")
    articles = fetch_pubmed_articles(topic)
    
    if articles:
        db = create_vector_db(articles)
        if db:
            report = ask_medical_bot(topic, db)
            
            print("\n" + "="*60)
            print("🔬 MEDICAL RESEARCH SUMMARY")
            print("="*60)
            print(report)
            print("="*60)
    else:
        print("No articles found for that topic.")