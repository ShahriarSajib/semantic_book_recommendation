import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.set_page_config(page_title="OSS Book Finder", page_icon="📚", layout="wide")
st.title("📚 Open Source Semantic Book Finder")

@st.cache_resource
def load_local_resources():
    # Tie lookups into the exact local configuration used during preprocessing
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory="./chromadb", embedding_function=embeddings)
    df = pd.read_csv("./data/cleaned_books.csv")
    return db, df

db, df = load_local_resources()

# Interface design layers
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    user_query = st.text_input("Enter concepts, topics, or book plot points:")
with col2:
    category_filter = st.selectbox("Category Facet Filter:", ["All", "fiction", "non-fiction"])
with col3:
    sort_tone = st.selectbox("Prioritize Story Vibe Tone:", ["Standard Match", "Joyful Tone", "Sad/Melancholic"])

if user_query:
    st.subheader("Results Engine Match Output")
    
    # Extract top results out of our vector database base layer
    raw_matches = db.similarity_search(user_query, k=25)
    
    # Parse vector document objects cleanly into a filterable pandas working dataframe 
    records = []
    for doc in raw_matches:
        title = doc.metadata.get('title')
        # Match back against your CSV columns to gather category/tone data
        csv_row = df[df['title'] == title]
        if not csv_row.empty:
            records.append(csv_row.iloc[0].to_dict())
            
    results_df = pd.DataFrame(records)
    
    if not results_df.empty:
        # Apply Zero-Shot Filter matching user choice
        if category_filter != "All":
            results_df = results_df[results_df['category'] == category_filter]
            
        # Re-sort display frames on calculated sentiment metadata columns
        if sort_tone == "Joyful Tone":
            results_df = results_df.sort_values(by='tone_joy', ascending=False)
        elif sort_tone == "Sad/Melancholic":
            results_df = results_df.sort_values(by='tone_sadness', ascending=False)
            
        # Display the matched records
        for idx, row in results_df.head(5).iterrows():
            with st.container(border=True):
                st.write(f"### {row['title']} — by *{row['authors']}*")
                st.caption(f"Tag Classification: **{row['category'].upper()}** | Rating: ⭐ {row.get('average_rating', 'N/A')}")
    else:
        st.write("No direct metadata matches found inside candidate constraints pool.")