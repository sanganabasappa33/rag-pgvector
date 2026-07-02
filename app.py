import streamlit as st

import tempfile

from ingest import process_pdf

from retriever import retrieve

from rag_chain import generate_answer


st.set_page_config(
    page_title="PDF RAG",
    layout="wide"
)

st.title(
    "PDF RAG with NeonDB + Groq"
)


uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if uploaded_file:

    if st.button("Process PDF"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            pdf_path = tmp.name

        with st.spinner(
            "Processing PDF..."
        ):

            count = process_pdf(
                pdf_path
            )

        st.success(
            f"{count} chunks stored in NeonDB"
        )


question = st.text_input(
    "Ask a question"
)

if question:

    docs = retrieve(question)

    context = "\n\n".join(
        docs
    )

    answer = generate_answer(
        question,
        context
    )

    st.subheader("Answer")

    st.write(answer)

    with st.expander(
        "Retrieved Context"
    ):

        for doc in docs:

            st.write(doc)
            st.divider()