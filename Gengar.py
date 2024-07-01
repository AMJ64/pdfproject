import os
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to retrieve relevant text from the PDF content using TF-IDF
def retrieve_relevant_text(pdf_text, query):
    # Split the PDF text into chunks (e.g., paragraphs)
    chunks = pdf_text.split('\n\n')
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit_transform(chunks + [query])
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity between the query and the chunks
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()
    
    # Get the most relevant chunk
    most_relevant_index = np.argmax(cosine_similarities)
    relevant_text = chunks[most_relevant_index]
    
    return relevant_text

# Function to save chat history to a file
def save_chat_history(chat_history, filename="chat_history.json"):
    with open(filename, "w") as file:
        json.dump(chat_history, file)

# Function to load chat history from a file
def load_chat_history(filename="chat_history.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return json.load(file)
    return []

# Function to clear chat history
def clear_chat_history(filename="chat_history.json"):
    if os.path.exists(filename):
        os.remove(filename)

# Streamlit UI
st.title("PDF Patola")
st.write("kya puchna he????")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Clear chat history when a new PDF is uploaded
    if "uploaded_file" in st.session_state and st.session_state.uploaded_file != uploaded_file:
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]
        clear_chat_history()
    
    st.session_state.uploaded_file = uploaded_file

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    # st.write("Extracted Text:")
    # st.write(pdf_text)

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()


    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(f"**prompt:** {message}")
        else:
            st.write(f"{message}")

    # Input box at the bottom
    user_input = st.text_area("Ask a question about the PDF content:", key="user_input_bottom", height=10)

    if st.button("Send", key="send_bottom"):
        if user_input:
            # Retrieve relevant text from the PDF content
            relevant_text = retrieve_relevant_text(pdf_text, user_input)
            
            # Generate response using the Groq client
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. The following is the relevant content of the PDF: " + relevant_text
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
                model="llama3-70b-8192",
            )
            response = chat_completion.choices[0].message.content
            
            # Update chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))
            
            # Save chat history
            save_chat_history(st.session_state.chat_history)
            
            # Clear the input box
            st.experimental_rerun()

    # Clear chat history button
    if st.button("Clear Chat History"):
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]
        clear_chat_history()
        st.experimental_rerun()
