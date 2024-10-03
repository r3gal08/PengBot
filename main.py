"""

TODO :
- Improve vectorDB
- Move history retrieval to a DB
- Ensure history is persisted over multiple chats (maybe even tied to a specific user?)
- Create a netdata/Grafana application to properly track resources when developing
- There must be a way to create better and more accurate vectorDBs...

"""

# Operational packages....
import ollama                                                               # API for LLM interaction
import streamlit as st                                                      # Web app framework
from langchain_community.vectorstores import FAISS                          # FAISS library for efficient similarity search
from langchain_community.embeddings import HuggingFaceEmbeddings            # Embeddings for document retrieval
from functions.chat_history import save_chat_history, display_chat_history  # Custom functions for handling chat history
from functions.load_api_token import load_api_token                         # Module for loading API token(s)
from pprint import pprint                                                   # Pretty print for debugging
import time                                                                 # Import the time module
from rich.console import Console                                            # Import Console from the Rich library

# # Uncomment to enable PDF and vectorDB creation:
#from functions.pdf_processing import process_pdf_and_vectordb
#selected_pdf = process_pdf_and_vectordb()
#print("Selected_pdf: " + selected_pdf + "\n")
selected_pdf = "NPPE-Syllabus"                          # Pass in pdf source name with extension stripped off
model_name = 'sentence-transformers/all-MiniLM-L6-v2'   # Pass in huggingface model to be used
console = Console()                                     # Create a Console object

# Use st.cache_resource to cache the model loading process. This is good for non-data objects that don't change often.
# st.cache_resource is also thread-safe. Therefore, it can be interacted with by multiple users safely
# If we wanted resources only to be available to a specific session (or user) Session state should be used instead
@st.cache_resource
def load_embeddings(model_name):
    print("\n*** Loading Hugging Face Embeddings Model for the first time ***\n")
    # Load the model and return it (it will be cached by Streamlit)
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'})

# TODO: I am not completely certain this will always be thread safe, But I think it is? (at least for now...)
# Cache the vector database (vectorDB) loading
# We disable parameter hashing here on the embeddings' argument. This is due to it being a complex argument.
# This is important to note mainly due to the fact that st.cache_resource will no longer re-run if it detects a
# change in the "embeddings" variable. THis should be fine in the context of this project, but worth noting if functionality ever changes
@st.cache_resource
def load_vectordb(selected_pdf, _embeddings):
    print("\n*** Loading VectorDB for the first time ***\n")
    return FAISS.load_local(f"vectordb/required_{selected_pdf}_vectordb", embeddings, allow_dangerous_deserialization=True)

# TODO: maybe move this to a centralized config.py file at some point...
# Load environment variables from .env file
try:
    load_api_token()        # Load the API token
except ValueError as e:     # Handle the error (e.g., log it or exit the program)
    print(e)

# Initialize an empty chat history if not already present in the session state
if "chat_history_local" not in st.session_state:
    st.session_state.chat_history_local = []
    # TODO: What more can be done here? Learn more about this initial instruction...
    st.session_state.llm_history_local = [      # LLM system context (initial instruction to the assistant)
        {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    ]

st.write("# PengBot")  # Display the app title
st.divider()           # Horizontal divider for UI separation

# TODO: Could add a user in here....
# Handle new chat or user message interaction logic here...
user_message = st.chat_input("You:", key="user_message")

if user_message:
    # Loading FAISS Database: Load the vector database corresponding to the selected PDF.
    #                         The database contains vector embeddings of document chunks.
    # What is FAISS?: FAISS (Facebook AI Similarity Search) allows efficient similarity search for large collections of vectors,
    #                 ideal for document embeddings and fast retrieval.

    # Load or reuse the cached embeddings model
    embeddings = load_embeddings(model_name)

    # Load or reuse the cached vectorDB
    db = load_vectordb(selected_pdf, embeddings)

    # Retrieving Documents: The retriever.invoke(user_message) method queries the FAISS database using the user's message
    #                       to find the most relevant sections of the PDF.

    retriever = db.as_retriever()           # Create a retriever from the FAISS database
    docs = retriever.invoke(user_message)   # Find relevant document chunks based on the user message

    # Retrieve the 2 most relevant document chunks to use as context for the LLM
    context = ""; count = 0
    for i in range(len(docs)):
        context += docs[i].page_content # Append the page content to the context
        count+=1
        if count==2: break  # Limit to 2 chunks

    # Debugging: Print the retrieved context
    # print("\n*** Context: ***")
    # pprint(context)
    # print("\n")

    # st.session_state.llm_history_local provides local history + what appears to be the index.pkl file created
    # when creating our vectordb. My assumption is this is bc we are abusing what "history" really is, but also this
    # .pkl file is only 3000 lines, so likely doesn't contain the entire book? TODO: Confirm thoughts here.
    st.session_state.chat_history_local.append({"You":user_message})
    st.session_state.llm_history_local.append({"role": "user", "content": context + "\n" + user_message})

    # Query ollama LLM with restful API
    stream = ollama.chat(
        model='llama3.1',
        messages=st.session_state.llm_history_local,
        stream=True,
    )

    # Build the assistant's response incrementally as chunks are received
    new_message = {"role": "assistant", "content": ""}
    print("\n\n*** New Bot Message Incoming ***\n")

    # Record the start time and re-init chars variable
    chars = 0
    start = time.time()

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)  # Print to console
        new_message["content"] += chunk['message']['content']   # Append to new_message array to display within streamlit app
        chars += len(chunk['message']['content'])               # Count characters

    # Stop the timer after the streaming is complete
    end = time.time()  # Record the end time

    # Print the results with formatting
    console.print(f"\n\n{'Time taken:':<12} {end-start:.2f} seconds", style="yellow")
    console.print(f"{'Chars:':<12} {chars/(end-start):.2f} /second", style="green")

    # Append the assistant's message to the chat history
    st.session_state.chat_history_local.append({"Bot":new_message["content"]})
    st.session_state.llm_history_local.append(new_message)

    # TODO: We eventually want this to be DB...
    # Save the chat history to a JSON file
    save_chat_history(st.session_state.chat_history_local, f"History/{st.session_state.chat_history_local[0]['You'].strip().strip('?')}.json")

# Display the entire chat history on the app interface
display_chat_history(st.session_state.chat_history_local)
