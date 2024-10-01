"""

TODO :
- I may be able to fix this annoying import error by adding my py environment to the "python packages" menu found in the bottom left.....
- Change this to "main.py" since its well, the main.....
- Version manage this bitch in github

"""

# Operational packages....
import streamlit as st                                                      # Web app framework
import glob                                                                 # File pattern matching (for PDFs)
import ollama                                                               # API for LLM interaction
from langchain_community.vectorstores import FAISS                          # FAISS library for efficient similarity search
from langchain_community.embeddings import HuggingFaceEmbeddings            # Embeddings for document retrieval
from functions.chat_history import save_chat_history, display_chat_history  # Custom functions for handling chat history
from functions.load_api_token import load_api_token                         # Module for loading api token(s)
from functions.pdf2text import required_txt                                 # Function to extract text from PDFs
from functions.vectordb import create_vectordb                              # Function to create a FAISS vector database

# Development packages....
from dotenv import load_dotenv  # Loads environment variables from a .env file for configuration
import os                       # OS-related functions
from pprint import pprint       # Pretty print for debugging

# TODO: maybe move this to a centralized config.py file at some point...
# TODO: Hugging face isn't even being used right now it seems....
# Load environment variables from .env file
try:
    load_api_token()  # Load the API token
except ValueError as e:
    print(e)  # Handle the error (e.g., log it or exit the program)

# Initialize an empty chat history if not already present in the session state
if "chat_history_local" not in st.session_state:
    st.session_state.chat_history_local = []
    # TODO: What more can be done here? Learn more about this initial instruction...
    st.session_state.llm_history_local = [      # LLM system context (initial instruction to the assistant)
        {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
                                   ]

st.write("# PengAI")    # Display the app title
st.divider()            # Horizontal divider for UI separation

# TODO: pdf and vectorDB creation is something I don't even really want this program to do (IE vectorDB should be static once its generated correctly),
#       Remove/change to a helper script at some point
# Select a textbook (PDF files in the 'Material' directory)
directory = 'Material'
pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
options = []
# Extract the PDF names without extension
for pdf_file in pdf_files:
    options.append(pdf_file.split("/")[-1][:-4])

# TODO: Similarly here: This program should assume the resource it is requiring, pdf conversion and vectorDB creation should be extracted into a cmdline utility/done outside of this
with st.sidebar:
    # Select textbook
    selected_pdf = st.selectbox("Book name: ", options)
    click1 = st.button(":orange[Create Requirements]")
    if click1: 
        st.write(":green[Creating required pdf]")
        required_txt(f"{selected_pdf}.pdf")
        st.write(":green[Creating vectordb]")
        create_vectordb(f"required_{selected_pdf}.pdf")
        st.write(":green[Requirements loaded]")
    st.divider()

    # Create button to start a New chat
    click2 = st.button(":orange[New Chat]")
    if click2:
        # Reset the chat history when starting a new chat
        st.session_state.chat_history_local = []
        st.session_state.llm_history_local = [
                                        {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
                                    ]
        display_chat_history(st.session_state.chat_history_local)

user_message = st.chat_input("You:", key="user_message")

if user_message:
    # TODO: Not actually sure if this is working...
    # Initialize Hugging Face embeddings (using a pre-trained model) for document retrieval
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cuda'})

    """
    Loading FAISS Database: Load the vector database corresponding to the selected PDF. 
                            The database contains vector embeddings of document chunks.
    What is FAISS?: FAISS (Facebook AI Similarity Search) allows efficient similarity search for large collections of vectors, 
                    ideal for document embeddings and fast retrieval.
    """
    # TODO: Consider optimizing by not loading the vector database on every user input
    print("\n*** (D)SelectedDB: " + selected_pdf + " ***\n")
    db = FAISS.load_local(f"vectordb/required_{selected_pdf}_vectordb", embeddings, allow_dangerous_deserialization=True)

    """
    Retrieving Documents: The retriever.invoke(user_message) method queries the FAISS database using the user's message 
                        to find the most relevant sections of the PDF.
    """
    retriever = db.as_retriever()           # Create a retriever from the FAISS database
    docs = retriever.invoke(user_message)   # Find relevant document chunks based on the user message

    # Retrieve the 2 most relevant document chunks to use as context for the LLM
    context = ""; count = 0
    for i in range(len(docs)):
        context += docs[i].page_content # Append the page content to the context
        count+=1
        if count==2: break  # Limit to 2 chunks

    # Debugging: Print the retrieved context
    print("\n*** Context: ***")
    pprint(context)
    print("\n")

    """
    st.session_state.llm_history_local provides local history + what appears to be the index.pkl file created
    when creating our vectordb. My assumption is this is bc we are abusing what "history" really is, but also this
    .pkl file is only 3000 lines, so likely doesn't contain the entire book? TODO: Confirm thoughts here.
    """
    st.session_state.chat_history_local.append({"You":user_message})
    st.session_state.llm_history_local.append({"role": "user", "content": context + "\n" + user_message})

    # TODO: Investigate additional features of the stream object
    # Query ollama LLM with restful API
    stream = ollama.chat(
        model='llama3.1',
        messages=st.session_state.llm_history_local,
        stream=True,
    )

    # Build the assistant's response incrementally as chunks are received
    new_message = {"role": "assistant", "content": ""}
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        print(chunk['message']['content'])
        new_message["content"] += chunk['message']['content']

    # Append the assistant's message to the chat history
    st.session_state.chat_history_local.append({"Bot":new_message["content"]})
    st.session_state.llm_history_local.append(new_message)

    # TODO: We eventually want this to be DB...
    # Save the chat history to a JSON file
    save_chat_history(st.session_state.chat_history_local, f"History/{st.session_state.chat_history_local[0]['You'].strip().strip('?')}.json")

# Display the entire chat history on the app interface
display_chat_history(st.session_state.chat_history_local)