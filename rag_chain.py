from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from config import GROQ_API_KEY


llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer only from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)


def generate_answer(
    question,
    context
):

    return chain.invoke(
        {
            "context": context,
            "question": question
        }
    )