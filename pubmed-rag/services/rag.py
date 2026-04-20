from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from services.prompt_template import template

def ask_medical_bot(question, vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    context_docs = retriever.invoke(question)

    context_text = "\n\n".join(
        [
            f"SOURCE: {d.metadata['title']} ({d.metadata['source_url']})\nCONTENT: {d.page_content}"
            for d in context_docs
        ]
    )

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    res = chain.invoke({"context": context_text, "question": question})

    return res.content if hasattr(res, "content") else str(res)