# Movie Recommendation System 🎬

A movie recommendation engine that combines data from TMDB and Letterboxd to provide personalized movie suggestions. Built with Python, utilizing a hybrid approach that considers content, director, and cast similarities.

## Overview

This project creates a comprehensive movie recommendation system by:
1. Scraping and combining movie data from TMDB API and Letterboxd reviews
2. Implementing a hybrid recommendation algorithm that weighs multiple similarity factors
3. Providing an interactive Streamlit interface for exploring movie recommendations

## Features

- **Data Collection Pipeline**
  - Comprehensive movie details from TMDB (plot, cast, crew, ratings, etc.)
  - User reviews and ratings from Letterboxd
  - Robust error handling and rate limiting for API requests
  - Automatic data cleaning and preprocessing

- **Hybrid Recommendation Engine**
  - Content-based similarity using TF-IDF vectorization
  - Director-based recommendations
  - Cast similarity analysis
  - Configurable weights for different similarity factors
  - Detailed similarity breakdown for recommendations

- **Interactive Web Interface**
  - Built with Streamlit for a responsive user experience
  - Adjustable recommendation weights via sidebar
  - Detailed movie information display
  - Expandable movie details and recommendation explanations

## Tech Stack

- **Data Collection**: Python, TMDB API, BeautifulSoup4, Requests
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Recommendation Engine**: TF-IDF, Cosine Similarity, MultiLabelBinarizer
- **Web Interface**: Streamlit
- **Additional Libraries**: SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movielens.git
cd movielens
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file in the project root
- Add your TMDB API key:
```
TMDB_API_KEY=your_api_key_here
```

## Usage

1. Scrape movie data:
```bash
python movieDataScraper.py
```

2. Run the Streamlit app:
```bash
streamlit run movieRecomApp.py
```

3. Use the interface:
- Select a movie from the dropdown
- Adjust similarity weights in the sidebar if desired
- Click "Get Recommendations" to see similar movies

## Data Structure

The project creates two main datasets:
- `movies.csv`: Comprehensive movie information including metadata, cast, crew, and technical details
- `reviews.csv`: User reviews and ratings scraped from Letterboxd


## Acknowledgments

- TMDB for providing the movie database API
- Letterboxd for the review data
- Streamlit for the web app framework