import os, json
from Bio import Entrez
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

NCBI_API_KEY = os.getenv("API_KEY")
Entrez.email = os.getenv("EMAIL_ID")

def fetch_pubmed_articles(query, max_results=10):
    print(f"Searching PubMed for: {query}...")

    try:
        handle = Entrez.esearch(
            db="pubmed", term=query, retmax=max_results, sort="pub_date"
        )
        record = Entrez.read(handle)
        id_list = record["IdList"]

        if not id_list:
            return []

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        articles = Entrez.read(handle)

        docs = []
        for article in articles["PubmedArticle"]:
            pmid = article["MedlineCitation"]["PMID"]
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]

            abstract = (
                article["MedlineCitation"]["Article"]
                .get("Abstract", {})
                .get("AbstractText", [])
            )
            abstract_text = " ".join(map(str, abstract))

            doc = Document(
                page_content=f"{title}\n{abstract_text}",
                metadata={
                    "title": title,
                    "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                },
            )
            docs.append(doc)
            print(f"Found: {title[:120]}...")
        return docs

    except Exception as e:
        print(f"PubMed Error: {e}")
        return []
