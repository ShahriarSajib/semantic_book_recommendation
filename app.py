import pandas as pd
import numpy as np
from dotenv import load_dotenv
import torch

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
load_dotenv()

# 2. Resource Caching for Free Local Models & Data Loading
@st.cache_resource
def load_data_and_vector_db():
    # Load dataset
    books = pd.read_csv("data/books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "assets/cover-not-found.jpg",
        books["large_thumbnail"],
    )
    
    # Set chunk_size=1 to satisfy modern LangChain rules while preserving line-by-line splits
    raw_documents = TextLoader("data/tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    
    # Auto-detect hardware for Windows machines (CUDA GPU or CPU fallback)
    device_target = "cuda" if torch.cuda.is_available() else "cpu"
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': device_target},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db_books = Chroma.from_documents(documents, hf_embeddings)
    return books, db_books

# Run the cached startup pipeline
books, db_books = load_data_and_vector_db()

# 3. Core Logic System Functions
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Strip any potential hidden quotation marks or trailing line breaks safely
    books_list = [int(rec.page_content.strip('"\n ').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = str(description).split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = str(row["authors"]).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"**{row['title']}**\n\n*by {authors_str}*\n\n{truncated_description}"
        results.append({"image": row["large_thumbnail"], "caption": caption})
    return results

# 4. Streamlit UI View Setup
st.title("📚 Semantic Book Recommender")

categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# UI Grid layout for filters and text inputs
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    user_query = st.text_input(
        label="Please enter a description of a book:",
        placeholder="e.g., A story about forgiveness"
    )
with col2:
    category_dropdown = st.selectbox("Select a category:", options=categories, index=0)
with col3:
    tone_dropdown = st.selectbox("Select an emotional tone:", options=tones, index=0)

submit_button = st.button("Find recommendations", type="primary")

st.markdown("---")
st.header("Recommendations")

# 5. UI Event Execution & Image Layout Grid
if submit_button:
    if not user_query.strip():
        st.warning("Please enter a query description to discover book recommendations.")
    else:
        with st.spinner("Searching local vector database..."):
            results = recommend_books(user_query, category_dropdown, tone_dropdown)
            
        if not results:
            st.info("No matching books found for selected filtering combinations.")
        else:
            # Replicate Gradio's Gallery layout using responsive row items (8 columns)
            MAX_COLUMNS = 8
            for i in range(0, len(results), MAX_COLUMNS):
                row_slice = results[i : i + MAX_COLUMNS]
                cols = st.columns(MAX_COLUMNS)
                
                for idx, item in enumerate(row_slice):
                    with cols[idx]:
                        st.image(item["image"], use_container_width=True)
                        st.markdown(item["caption"])