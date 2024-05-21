import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data(url):
    data_movies = pd.read_csv(url)
    for feature in ['MovieName', 'Genre', 'Actor', 'Director']:
        data_movies[feature] = data_movies[feature].fillna('No information available')
    return data_movies

def movie_recomm(user_movieName, all_titles_list, data_movies, cos_similar):
    search_close_match = difflib.get_close_matches(user_movieName, all_titles_list, n=5)
    if not search_close_match:
        st.error("No close matches found! Please check your input and try a different movie.")
        return None
    selected_movie = st.selectbox('Did you mean:', search_close_match, key='selected_movie')
    if st.button('Yes, recommend!', key='confirm_recommend'):
        return selected_movie
    return None

def show_recommendations(selected_movie, data_movies, cos_similar):
    if selected_movie:
        index_movie = data_movies[data_movies.MovieName == selected_movie].index.values[0]
        similar_score = list(enumerate(cos_similar[index_movie]))
        sorted_similar_mov = sorted(similar_score, key=lambda x: x[1], reverse=True)
        st.subheader("Suggesting similar movies for you:")
        for movie in sorted_similar_mov[1:21]:  # Skip the first as it is the movie itself with 100% similarity
            index = movie[0]
            similarity_percentage = movie[1] * 100  # Convert fraction to percentage
            title_fr_index = data_movies.at[index, "MovieName"]
            st.write(f"{title_fr_index} - Similarity: {similarity_percentage:.2f}%")

def main():
    st.set_page_config(page_title="Tamil Movie Recommender", page_icon=":movie_camera:", layout="wide")

    st.markdown("""
    <style>
    body {
        color: #0000ff;
        background-color: #ff0000;
        font-family: 'Helvetica';
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .img-slider {
        display: flex;
        overflow-x: auto;
        animation: slide 5s infinite linear;
    }
    @keyframes slide {
        from { transform: translateX(0); }
        to { transform: translateX(-33.33%); }
    }
    img {
        height: 350px;  # Adjust height as needed
        margin-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Image Slider
    st.markdown("""
    <div class="img-slider">
        <img src="https://raw.githubusercontent.com/AnanyaThyagarajan/Movie-Recomm/main/109799_thumb_665.jpg" alt="Tamil Movies 1">
        <img src="https://raw.githubusercontent.com/AnanyaThyagarajan/Movie-Recomm/main/94070_thumb_665.jpg" alt="Tamil Movies 2">
        <img src="https://raw.githubusercontent.com/AnanyaThyagarajan/Movie-Recomm/main/Tamil-Movies-1-1.jpg" alt="Tamil Movies 3">
    </div>
    """, unsafe_allow_html=True)

    st.title('Tamil Movie Recommender')
    st.markdown('## Discover Movies Similar to Your Favorites!')

    data_movies = load_data('https://raw.githubusercontent.com/AnanyaThyagarajan/Python-Projects/main/Movie%20Recommend/Tamil_movies_dataset.csv')
    all_titles_list = set(data_movies['MovieName'].tolist())
    feature_combined = data_movies['MovieName'] + " " + data_movies['Genre'] + " " + data_movies['Actor'] + " " + data_movies['Director']
    tf_vector = TfidfVectorizer()
    feature_vect = tf_vector.fit_transform(feature_combined)
    cos_similar = cosine_similarity(feature_vect)

    user_movieName = st.text_input("Enter the name of your favorite movie", key="movie_input")
    if st.button('Recommend', key='get_recommend'):
        selected_movie = movie_recomm(user_movieName, all_titles_list, data_movies, cos_similar)
        elif selected_movie:
            show_recommendations(selected_movie, data_movies, cos_similar)
       

if __name__ == "__main__":
    main()

