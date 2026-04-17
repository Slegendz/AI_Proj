import os
from Bio import Entrez
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from prompt import medical_prompt

load_dotenv()

# 1. SETUP
Entrez.email = os.environ.get("EMAIL_ID")
Entrez.api_key = os.environ.get("API_KEY")

# Use 'nomic-embed-text' for the vector math (ollama pull nomic-embed-text)
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
embeddings = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# This creates a folder 'pubmed_db' to store your data permanently
vectorstore = Chroma(persist_directory="./pubmed_db", embedding_function=embeddings)
# llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)
llm = ChatOllama(model=LANGUAGE_MODEL, temperature=0)

def fetch_and_add_to_db(query):
    """Search PubMed and save results to our local database"""
    print(f"Searching PubMed for: {query}...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
    record = Entrez.read(handle)
    ids = record["IdList"]
    
    fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
    articles = Entrez.read(fetch_handle)
    
    new_docs = []
    
    a = 0
    for article in articles['PubmedArticle']:
        if(a == 0):
            print(article)
            a += 1
        
        title = article['MedlineCitation']['Article']['ArticleTitle']
        pmid = str(article['MedlineCitation']['PMID'])

        # Extract abstract
        abstract = ""
        if 'Abstract' in article['MedlineCitation']['Article']:
            abstract = " ".join(article['MedlineCitation']['Article']['Abstract']['AbstractText'])
        
        # Create a document with metadata for citations
        doc = Document(
            page_content=f"Title: {title}\nAbstract: {abstract}",
            metadata={"source": f"PMID: {pmid}", "title": title}
        )
        new_docs.append(doc)
    
    # Add to ChromaDB
    if new_docs:
        vectorstore.add_documents(new_docs)
        print(f"Added {len(new_docs)} new articles to memory.")

# 2. THE LOOP
while True:
    user_query = input("\nAsk a medical question (or type 'bye'): ")
    if user_query.lower() == 'bye':
        break
    
    # Step 1: "Train" by fetching new data related to the question
    fetch_and_add_to_db(user_query)
    
    # Step 2: Use RAG to answer
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm, 
    #     retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    #     return_source_documents=True
    # )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": medical_prompt}
    )
    
    response = qa_chain.invoke({"query": user_query})
    
    print("\n--- ANSWER ---")
    print(response["result"])
    print("\n--- SOURCES ---")

    for doc in response["source_documents"]:
        print(f"- {doc.metadata['title']} ({doc.metadata['source']})")