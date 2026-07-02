from sqlalchemy import create_engine
from sqlalchemy import text

from config import DATABASE_URL

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

engine = create_engine(
    DATABASE_URL
)


def retrieve(question):

    query_embedding = (
        embedding_model.embed_query(question)
    )

    sql = """
    SELECT content
    FROM pdf_chunks

    ORDER BY embedding <=> :embedding

    LIMIT 3
    """

    with engine.connect() as conn:

        rows = conn.execute(
            text(sql),
            {
                "embedding": str(query_embedding)
            }
        ).fetchall()

    return [row[0] for row in rows]