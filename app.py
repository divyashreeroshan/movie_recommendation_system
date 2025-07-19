import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df['genres'] = df['genres'].replace("(no genres listed)", "")
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("⚙️ Filters & Settings")

# Theme toggle
theme = st.sidebar.radio("Select Theme", ("Light", "Dark"))
st.markdown(
    f"<style>body {{ background-color: {'#f0f2f6' if theme == 'Light' else '#0e1117'}; color: {'#000' if theme == 'Light' else '#fff'} }}</style>",
    unsafe_allow_html=True
)

# Genre filter
all_genres = sorted(set(g for genre_list in df['genres'] for g in genre_list.split('|')))
selected_genre = st.sidebar.selectbox("🎭 Filter by Genre (optional)", ["All"] + all_genres)

# Number of recommendations
num_recs = st.sidebar.slider("📽️ Number of Recommendations", 5, 20, 10)

# --- TF-IDF on genres ---
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# --- Nearest Neighbors Model ---
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# --- Title to Index Mapping ---
title_to_index = pd.Series(df.index, index=df['title']).drop_duplicates()

# --- Recommendation Function ---
def get_recommendations(title, num_recommendations=10):
    if title not in title_to_index:
        return pd.DataFrame()

    idx = title_to_index[title]
    movie_vector = tfidf_matrix[idx]
    distances, indices = nn_model.kneighbors(movie_vector, n_neighbors=num_recommendations + 1)
    recommended_indices = indices.flatten()[1:]  # skip the movie itself
    recommended_df = df.iloc[recommended_indices].copy()

    # Filter by genre if selected
    if selected_genre != "All":
        recommended_df = recommended_df[recommended_df['genres'].str.contains(selected_genre)]

    return recommended_df[['title', 'genres', 'poster_url'] if 'poster_url' in df.columns else ['title', 'genres']]

# --- UI Header ---
st.set_page_config(page_title="Movie Recommendation", page_icon="🎬", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Find movies similar to your favorite using genre similarity.</p>",
    unsafe_allow_html=True
)

# --- Movie Selection ---
selected_movie = st.selectbox("🎞️ Choose a movie:", df['title'].sort_values().unique())

# --- Recommend Button ---
if st.button("🚀 Get Recommendations"):
    results = get_recommendations(selected_movie, num_recommendations=num_recs)

    if results.empty:
        st.warning("⚠️ No recommendations found.")
    else:
        st.success(f"Movies similar to **{selected_movie}**:")
        for idx, row in results.iterrows():
            # Show poster if available
            if 'poster_url' in row and pd.notna(row['poster_url']):
                st.image(row['poster_url'], width=150, caption=row['title'])
            st.write(f"**{row['title']}** — _{row['genres']}_")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
