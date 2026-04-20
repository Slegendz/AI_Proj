# PubMed RAG: AI Medical Research Assistant 🧬

A Retrieval-Augmented Generation (RAG) pipeline that fetches the latest medical research from PubMed and uses Google's Gemini LLM to provide structured, evidence-based summaries.

## 🚀 Overview
This project addresses the challenge of keeping up with rapidly evolving medical literature. By connecting the **NCBI PubMed API** with **Google Gemini 1.5 Flash**, the system retrieves real-time abstracts, indexes them in a local vector database, and generates clinical summaries grounded in peer-reviewed data.

## 🛠️ Key Features
- **Real-time Retrieval:** No reliance on outdated training data; fetches current papers via Biopython.
- **Semantic Search:** Uses `text-embedding-004` to find relevant context even if keywords don't match exactly.
- **Precision Grounding:** Prevents AI hallucinations by forcing the LLM to answer using only retrieved abstracts.
- **Clean Output:** Custom parsing logic to remove API metadata and "signature" artifacts for a professional report.
- **Manual Batching:** Optimized for Google AI Free Tier to prevent rate-limit crashes.

## 🏗️ Technical Architecture


1. **Ingestion:** Fetches XML data from PubMed based on user query.
2. **Processing:** Splits text into 1000-character chunks with a 100-character overlap for context continuity.
3. **Indexing:** Converts text to vectors and stores them in a **ChromaDB** collection.
4. **Generation:** Retrieves the Top-3 most relevant chunks and feeds them to Gemini 1.5 Flash.

## 📋 Prerequisites
- Python 3.10+
- Google AI Studio [API Key](https://aistudio.google.com/)
- NCBI [API Key](https://pubmed.ncbi.nlm.nih.gov/settings/key/) (Required for stable PubMed access)

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Slegendz/AI_Proj.git](https://github.com/Slegendz/AI_Proj.git)
   cd AI_Proj/pubmed-rag