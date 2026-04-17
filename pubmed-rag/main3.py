import os
from Bio import Entrez
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Loading secrets
load_dotenv()

# Configurations
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
Entrez.email = os.environ.get("EMAIL_ID")
NCBI_API_KEY = os.environ.get("API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")

# Fetching PubMed articles
def fetch_pubmed_articles(query, max_results=4):
    print(f"Searching PubMed for: {query}...")

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="pub_date")
        record = Entrez.read(handle)
        id_list = record["IdList"]
        
        # Returns empty list if no articles found
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
            print(f"Found: {title[:70]}...")
        return docs
    except Exception as e:
            print(f"PubMed Error: {e}")
            return []
    
def create_vector_db(docs):
    print("Processing text and generating embeddings...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    if not splits:
        print("No text splits found.")
        return None

    try:
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name="pubmed_research",
        )
        return vectorstore
    except Exception as e:
        print(f"Error during embedding: {e}")
        return None
    
    
# Using gemini model 
def ask_medical_bot(question, vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.invoke(question)
    
    context_text = "\n\n".join([
        f"SOURCE: {d.metadata['title']} ({d.metadata['source_url']})\nCONTENT: {d.page_content}" 
        for d in context_docs
    ])

    template = """
    You are a professional Medical Research Assistant. 
    Use the following pieces of retrieved context to answer the question.
    
    REQUIRED FORMAT:
    1. Summary: Concise 2-3 sentence overview.
    2. Detailed Findings: Bullet points.
    3. References: List Title Only and its ID.

    CONTEXT:
    {context}

    QUESTION: {question}
    YOUR RESPONSE (Keep it brief and professional):
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({"context": context_text, "question": question})

    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list):
            return content[0].get('text', 'No content found.')
        return str(content)
    
    return "Error: Could not retrieve clean response."

if __name__ == "__main__":
    topic = input("💬 Topic: ")
    articles = fetch_pubmed_articles(topic)

    if articles:
        db = create_vector_db(articles)
        if db:
            raw_response = ask_medical_bot(topic, db)
            
            print("\n" + "="*60)
            print("🔬 MEDICAL RESEARCH REPORT")
            print("="*60)
            
            if hasattr(raw_response, 'content'):
                print(raw_response.content)
            else:
                print(raw_response)
                
            print("="*60)