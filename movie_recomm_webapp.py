import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Tamil Movie Recommender", page_icon=":movie_camera:", layout="wide")
# Load the data
@st.cache_data
def load_data(url):
    data_movies = pd.read_csv(url)
    for feature in ['MovieName', 'Genre', 'Actor', 'Director']:
        data_movies[feature] = data_movies[feature].fillna('No information available')
    return data_movies

def find_similar_movies(selected_movie, data_movies, cos_similar):
    index_movie = data_movies[data_movies.MovieName == selected_movie].index.values[0]
    similar_score = list(enumerate(cos_similar[index_movie]))
    sorted_similar_mov = sorted(similar_score, key=lambda x: x[1], reverse=True)
    return sorted_similar_mov[1:21]  # Skip the self-match


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

    data_url = 'https://raw.githubusercontent.com/AnanyaThyagarajan/Python-Projects/main/Movie%20Recommend/Tamil_movies_dataset.csv'
    data_movies = load_data(data_url)
    all_titles_list = set(data_movies['MovieName'].tolist())

    feature_combined = data_movies['MovieName'] + " " + data_movies['Genre'] + " " + data_movies['Actor'] + " " + data_movies['Director']
    tf_vector = TfidfVectorizer()
    feature_vect = tf_vector.fit_transform(feature_combined)
    cos_similar = cosine_similarity(feature_vect)

    user_movieName = st.text_input("Enter the name of your favorite movie", key="movie_input")

    if 'selected_movie' not in st.session_state:
        st.session_state['selected_movie'] = None

    if st.button('Recommend'):
        matches = difflib.get_close_matches(user_movieName, all_titles_list, n=5)
        if not matches:
            st.error("No close matches found. Please check your input and try again.")
        else:
            st.session_state['selected_movie'] = st.selectbox('Did you mean:', matches, key='movie_select')

    if st.session_state['selected_movie'] and st.button('Yes, recommend!', key='confirm_recommend'):
        recommendations = find_similar_movies(st.session_state['selected_movie'], data_movies, cos_similar)
        st.subheader("Suggesting similar movies for you:")
        for idx, (index, similarity) in enumerate(recommendations):
            movie_title = data_movies.at[index, "MovieName"]
            similarity_percentage = similarity * 100
            st.write(f"{idx+1}. {movie_title} - Similarity: {similarity_percentage:.2f}%")

if __name__ == "__main__":
    main()
