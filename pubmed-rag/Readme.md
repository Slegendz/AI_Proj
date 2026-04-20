# RAG (Retreival Augmented Generation)

Retreive => Get data from a source (e.g. Google, Wikipedia, PubMed, etc.)
Augment => Use context from the retrieved data to generate a response (e.g. using LLM)
Generate => Generate a response based on the retrieved data and the context provided.

## How RAG works

Components: Knowledge Source -> Text Chunking -> Embedding Model -> Vector Database -> Query Encoder -> Retriever -> Augmenter -> LLM (Generator) -> Updater

![RAG Workflow](https://media.geeksforgeeks.org/wp-content/uploads/20250210190608027719/How-Rag-works.webp)

1. External Knowledge Source: Stores domain specific or general information like documents, APIs or databases.
2. Text Chunking and Preprocessing: Breaks large text into smaller, manageable chunks and cleans it for consistency.
3. Embedding Model: Converts text into numerical vectors that capture semantic meaning.
4. Vector Database: Stores embeddings and enables similarity search for fast information retrieval.
5. Query Encoder: Transforms the user’s query into a vector for comparison with stored embeddings.
6. Retriever: Finds and returns the most relevant chunks from the database based on query similarity.
7. Prompt Augmentation Layer: Combines retrieved chunks with the user’s query to provide context to the LLM.
8. LLM (Generator): Generates a grounded response using both the query and retrieved knowledge.
9. Updater (Optional): Regularly refreshes and re-embeds data to keep the knowledge base up to date
