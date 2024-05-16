import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
@st.cache_data # This decorator caches the data to avoid reloading it every time the app runs
def load_data(url):
    data_movies = pd.read_csv(url)
    # Fill missing values with 'No information available'
    unique_features = ['MovieName', 'Genre', 'Actor', 'Director']
    for feature in unique_features:
        if data_movies[feature].isna().sum() > 0:
            data_movies[feature] = data_movies[feature].fillna('No information available')
    return data_movies

def movie_recomm(user_movieName, all_titles_list, data_movies, cos_similar):
    search_close_match = difflib.get_close_matches(user_movieName, all_titles_list, n=5)
    if not search_close_match:
        st.write("No close matches found! Please check your input and either try again or try a different movie.")
    else:
        close_match = st.selectbox('Did you mean:', search_close_match)
        if st.button('Confirm'):
            index_movie = data_movies[data_movies.MovieName == close_match].index.values[0]
            similar_score = list(enumerate(cos_similar[index_movie]))
            sorted_similar_mov = sorted(similar_score, key=lambda x: x[1], reverse=True)

            st.write("Suggesting similar movies for you:")
            for movie in sorted_similar_mov[1:21]:  # Skip the first as it is the movie itself with 100% similarity
                index = movie[0]
                similarity_percentage = movie[1] * 100  # Convert fraction to percentage
                title_fr_index = data_movies.at[index, "MovieName"]
                st.write(f"{title_fr_index} - Similarity: {similarity_percentage:.2f}%")

def main():
    st.title('Tamil Movie Recommender')
    url = 'https://raw.githubusercontent.com/AnanyaThyagarajan/Python-Projects/main/Movie%20Recommend/Tamil_movies_dataset.csv'
    data_movies = load_data(url)
    
    all_titles_list = set(data_movies['MovieName'].tolist())
    feature_combined = data_movies['MovieName'] + " " + data_movies['Genre'] + " " + data_movies['Actor'] + " " + data_movies['Director']
    tf_vector = TfidfVectorizer()
    feature_vect = tf_vector.fit_transform(feature_combined)
    cos_similar = cosine_similarity(feature_vect)
    
    user_movieName = st.text_input("Enter the name of your favorite movie")
    if user_movieName:
        movie_recomm(user_movieName, all_titles_list, data_movies, cos_similar)

if __name__ == "__main__":
    main()




