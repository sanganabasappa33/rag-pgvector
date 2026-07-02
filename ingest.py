from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from sqlalchemy import create_engine
from sqlalchemy import text

from config import DATABASE_URL


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    clean_docs = []

    for doc in documents:

        text_content = doc.page_content

        text_content = text_content.replace(
            "\n",
            " "
        )

        text_content = " ".join(
            text_content.split()
        )

        doc.page_content = text_content

        clean_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        clean_docs
    )

    engine = create_engine(
        DATABASE_URL
    )

    with engine.connect() as conn:

        for chunk in chunks:

            embedding = embedding_model.embed_query(
                chunk.page_content
            )

            conn.execute(
                text(
                    """
                    INSERT INTO pdf_chunks
                    (content, embedding)

                    VALUES
                    (:content, :embedding)
                    """
                ),
                {
                    "content": chunk.page_content,
                    "embedding": str(embedding)
                }
            )

        conn.commit()

    return len(chunks)