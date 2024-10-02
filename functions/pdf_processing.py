import os
import glob
from functions.pdf2text import required_txt  # Function to extract text from PDFs
from functions.vectordb import create_vectordb  # Function to create FAISS vector database
import streamlit as st  # For Streamlit UI components

# Function to process PDF and create vectorDB
def process_pdf_and_vectordb(directory='Material'):
    # Find all PDF files in the specified directory
    pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
    options = [pdf_file.split("/")[-1][:-4] for pdf_file in pdf_files]  # Extract file names without extension

    # Streamlit Sidebar for PDF selection and vectorDB creation
    with st.sidebar:
        # Select textbook (PDF)
        selected_pdf = st.selectbox("Book name: ", options)
        click1 = st.button(":orange[Create Requirements]")

        if click1:
            st.write(":green[Creating required pdf]")
            required_txt(f"{selected_pdf}.pdf")  # Extract text from the PDF
            st.write(":green[Creating vectordb]")
            create_vectordb(f"required_{selected_pdf}.pdf")  # Create the vectorDB for the PDF
            st.write(":green[Requirements loaded]")

        st.divider()

        return selected_pdf  # Return the selected PDF for further use
