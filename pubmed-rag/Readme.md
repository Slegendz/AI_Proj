# Pubmed RAG

To run the ollama qwen : ollama run qwen2.5:0.5b  (/bye for leaving)

Libraries installed: pip install langchain langchain-community langchain-ollama chromadb biopython

## How RAG WORKS 

In a RAG system, you use two different models:

The Embedding Model (nomic-embed-text): This is a specialized, small model that turns text into lists of numbers (vectors). It's used to "read" the PubMed articles and store them in ChromaDB.

The LLM (qwen): This is the "brain" that reads the search results and writes the final answer for you.
