import os
from Bio import Entrez
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP
Entrez.email = os.environ.get("EMAIL_ID")
NCBI_API_KEY  = os.environ.get("API_KEY")
LLM_MODEL = "qwen2.5:0.5b"
# EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")

# --- STEP 1: FETCH DATA FROM PUBMED ---
def fetch_pubmed_articles(query, max_results=8):
    print(f"🔎 Searching PubMed for: {query}...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="pub_date", api_key=NCBI_API_KEY)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    
    if not id_list:
        return []

    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml", api_key=NCBI_API_KEY)
    articles = Entrez.read(handle)
    
    docs = []
    for article in articles['PubmedArticle']:
        pmid = article['MedlineCitation']['PMID']
        title = article['MedlineCitation']['Article']['ArticleTitle']
        
        # Extract abstract text safely
        abstract_parts = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
        abstract_text = " ".join([str(p) for p in abstract_parts])
        
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        # Store metadata for citations
        docs.append(Document(
            page_content=f"Title: {title}\nAbstract: {abstract_text}",
            metadata={"source_url": url, "title": title, "pmid": str(pmid)}
        ))
        print(docs, sep = '\n')
    return docs

# --- STEP 2: PROCESS & STORE IN VECTOR DB ---
def create_vector_db(docs):
    print("🧠 Processing articles and creating vector index...")
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Store in ephemeral ChromaDB (clears every run for "newest" articles)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- STEP 3: STRUCTURED PROMPT & QUERY ---
def ask_medical_bot(question, vectorstore):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    
    # Retrieve top chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.invoke(question)
    
    context_text = "\n\n".join([
        f"SOURCE: {d.metadata['title']} (URL: {d.metadata['source_url']})\nCONTENT: {d.page_content}" 
        for d in context_docs
    ])

    template = """
    You are a professional Medical Research Assistant. Answer the question using ONLY the provided PubMed contexts.
    
    STRUCTURE YOUR OUTPUT AS FOLLOWS:
    1. Summary: A brief 2-3 sentence overview.
    2. Key Findings: Bullet points of specific data or discoveries.
    3. Citations: List the Titles and clickable URLs of the articles used.

    CONTEXT:
    {context}

    QUESTION: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    response = chain.invoke({"context": context_text, "question": question})
    return response.content

# --- MAIN WORKFLOW ---
if __name__ == "__main__":
    user_query = input("Enter your medical research topic: ")
    
    # Workflow Execution
    articles = fetch_pubmed_articles(user_query)
    
    if articles:
        db = create_vector_db(articles)
        answer = ask_medical_bot(user_query, db)
        print(answer)
        print("="*50)
    else:
        print("No recent articles found for that query.")