import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIR = "./chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def create_or_load_db(topic, docs = None):
    print("Processing text and generating embeddings...")

    safe_topic = topic.replace(" ", "_").lower()
    persist_dir = os.path.join(PERSIST_DIR, safe_topic)

    if os.path.exists(persist_dir):
        print(f"Loading existing DB for topic: {topic}")

        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="pubmed_research"
        )
        
    if not docs:
        print("No documents provided to create DB")
        return None

    print(f"Creating new DB for topic: {topic}")

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
            persist_directory= PERSIST_DIR
        )
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        print(f"Error during embedding: {e}")
        return None