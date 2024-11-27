import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import implicit
from typing import List, Tuple, Dict

class HybridRecommender:
    """
    Enhanced hybrid recommendation system that considers movie content, directors, and cast.
    """
    def __init__(self, 
                 content_weight: float = 0.4,
                 director_weight: float = 0.3,
                 cast_weight: float = 0.3):
        self.content_weight = content_weight
        self.director_weight = director_weight
        self.cast_weight = cast_weight
        self.content_similarities = None
        self.director_similarities = None
        self.cast_similarities = None
        self.item_features = None
        self.items_df = None
        self.mlb_cast = MultiLabelBinarizer()
        self.mlb_director = MultiLabelBinarizer()

    def _preprocess_text_features(self, items_df: pd.DataFrame) -> None:
        """Process text features using TF-IDF"""
        text_features = (
            items_df['description'].fillna('') + ' ' + 
            items_df['categories'].fillna('') + ' ' + 
            items_df['tags'].fillna('')
        )
        
        tfidf = TfidfVectorizer(max_features=5000)
        self.item_features = tfidf.fit_transform(text_features)

    def _preprocess_cast(self, items_df: pd.DataFrame) -> None:
        """Process cast members using one-hot encoding"""
        # Convert string of cast members to list
        cast_lists = items_df['cast'].fillna('').apply(
            lambda x: [name.strip() for name in str(x).split(',') if name.strip()]
        )
        self.cast_features = self.mlb_cast.fit_transform(cast_lists)

    def _preprocess_directors(self, items_df: pd.DataFrame) -> None:
        """Process directors using one-hot encoding"""
        # Convert string of directors to list
        director_lists = items_df['director'].fillna('').apply(
            lambda x: [name.strip() for name in str(x).split(',') if name.strip()]
        )
        self.director_features = self.mlb_director.fit_transform(director_lists)

    def preprocess_data(self, items_df: pd.DataFrame) -> None:
        """
        Preprocess all movie features including content, cast, and directors
        """
        self.items_df = items_df
        
        # Process different types of features
        self._preprocess_text_features(items_df)
        self._preprocess_cast(items_df)
        self._preprocess_directors(items_df)

    def train_models(self) -> None:
        """Calculate similarity matrices for all feature types"""
        # Calculate content-based similarities
        self.content_similarities = cosine_similarity(self.item_features)
        
        # Calculate cast similarities
        self.cast_similarities = cosine_similarity(self.cast_features)
        
        # Calculate director similarities
        self.director_similarities = cosine_similarity(self.director_features)

    def get_similar_movies(self, movie_title: str, n_recommendations: int = 5) -> List[Dict]:
        """
        Get similar movies based on weighted combination of content, cast, and director similarities
        """
        # Find the movie index
        movie_idx = self.items_df[self.items_df['title'].str.lower() == movie_title.lower()].index
        if len(movie_idx) == 0:
            return []
        
        movie_idx = movie_idx[0]
        
        # Calculate weighted similarities
        combined_similarities = (
            self.content_weight * self.content_similarities[movie_idx] +
            self.cast_weight * self.cast_similarities[movie_idx] +
            self.director_weight * self.director_similarities[movie_idx]
        )
        
        # Get top similar movies
        similar_indices = np.argsort(combined_similarities)[::-1][1:n_recommendations+1]
        
        # Format recommendations with detailed information
        recommendations = []
        for idx in similar_indices:
            item_data = self.items_df.iloc[idx]
            recommendations.append({
                'title': item_data['title'],
                'director': item_data['director'],
                'cast': item_data['cast'],
                'categories': item_data['categories'],
                'similarity_score': float(combined_similarities[idx]),
                'description': item_data['description'],
                'matching_factors': self._get_matching_factors(movie_idx, idx)
            })
            
        return recommendations

    def _get_matching_factors(self, movie_idx1: int, movie_idx2: int) -> Dict[str, float]:
        """Calculate individual similarity scores for each factor"""
        return {
            'content_similarity': float(self.content_similarities[movie_idx1, movie_idx2]),
            'cast_similarity': float(self.cast_similarities[movie_idx1, movie_idx2]),
            'director_similarity': float(self.director_similarities[movie_idx1, movie_idx2])
        }

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Get movie recommendations based on content, directors, and cast!")

    # Load data
    try:
        movies_df = pd.read_csv('data/movies.csv')
    except FileNotFoundError:
        st.error("Error: Required data file 'movies.csv' not found.")
        return

    # Add sidebar for weights configuration
    st.sidebar.header("Recommendation Weights")
    content_weight = st.sidebar.slider("Content Similarity Weight", 0.0, 1.0, 0.4, 0.1)
    director_weight = st.sidebar.slider("Director Similarity Weight", 0.0, 1.0, 0.3, 0.1)
    cast_weight = st.sidebar.slider("Cast Similarity Weight", 0.0, 1.0, 0.3, 0.1)

    # Normalize weights
    total_weight = content_weight + director_weight + cast_weight
    content_weight /= total_weight
    director_weight /= total_weight
    cast_weight /= total_weight

    # Initialize recommender
    @st.cache_resource
    def load_recommender(c_weight, d_weight, cast_weight):
        recommender = HybridRecommender(
            content_weight=c_weight,
            director_weight=d_weight,
            cast_weight=cast_weight
        )
        recommender.preprocess_data(movies_df)
        recommender.train_models()
        return recommender

    recommender = load_recommender(content_weight, director_weight, cast_weight)

    # Create a dropdown with movie titles
    movie_titles = sorted(movies_df['title'].tolist())
    selected_movie = st.selectbox(
        "Select a movie:",
        options=movie_titles
    )

    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            recommendations = recommender.get_similar_movies(selected_movie)
            
            if recommendations:
                # Display selected movie details
                selected_movie_data = movies_df[movies_df['title'] == selected_movie].iloc[0]
                st.subheader("Selected Movie Details:")
                with st.expander("Show Details"):
                    st.write(f"**Director:** {selected_movie_data['director']}")
                    st.write(f"**Cast:** {selected_movie_data['cast']}")
                    st.write(f"**Categories:** {selected_movie_data['categories']}")
                    st.write(f"**Description:** {selected_movie_data['description']}")

                # Display recommendations
                st.subheader(f"Top 5 movies similar to '{selected_movie}':")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['title']} (Overall Similarity: {rec['similarity_score']:.2f})"):
                        st.write(f"**Director:** {rec['director']}")
                        st.write(f"**Cast:** {rec['cast']}")
                        st.write(f"**Categories:** {rec['categories']}")
                        st.write(f"**Description:** {rec['description']}")
                        
                        # Show similarity breakdown
                        st.write("\n**Similarity Breakdown:**")
                        factors = rec['matching_factors']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Content", f"{factors['content_similarity']:.2f}")
                        with col2:
                            st.metric("Director", f"{factors['director_similarity']:.2f}")
                        with col3:
                            st.metric("Cast", f"{factors['cast_similarity']:.2f}")
            else:
                st.error("Movie not found in the database. Please try another title.")

if __name__ == "__main__":
    main()