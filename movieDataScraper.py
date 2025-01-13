import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import re

class EnhancedMovieScraper:
    """
    Enhanced scraper that collects comprehensive movie data from TMDB and Letterboxd,
    specifically designed for building a rich recommendation system dataset.
    """
    
    def __init__(self):
        load_dotenv()
        
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        if not self.tmdb_api_key:
            raise ValueError("TMDB API key not found! Make sure it's set in your .env file.")
        
        self.movies_data = []
        self.reviews_data = []
        
        # Setup logging
        logging.basicConfig(
            filename='scraping.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Create data folder
        Path('data').mkdir(exist_ok=True)
        
        # Cache for genre mappings from TMDB
        self.genre_mapping = self._get_genre_mapping()
        
    def _get_genre_mapping(self):
        """Fetch genre ID to name mapping from TMDB"""
        url = "https://api.themoviedb.org/3/genre/movie/list"
        params = {'api_key': self.tmdb_api_key}
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                genres = response.json()['genres']
                return {genre['id']: genre['name'] for genre in genres}
        except Exception as e:
            logging.error(f"Error fetching genre mapping: {str(e)}")
            return {}
        
    def get_full_movie_details(self, movie_id):
        """Fetch comprehensive movie details from TMDB including credits and keywords"""
        movie_details = {}
        
        # Get basic movie info
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            'api_key': self.tmdb_api_key,
            'append_to_response': 'credits,keywords,recommendations'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Get director(s)
                directors = [person['name'] for person in data['credits']['crew'] 
                           if person['job'] == 'Director']
                
                # Get top cast members (limit to top 10)
                cast = [person['name'] for person in data['credits']['cast'][:10]]
                
                # Get keywords/tags
                tags = [keyword['name'] for keyword in data['keywords']['keywords']]
                
                # Get production companies
                production_companies = [company['name'] for company in data['production_companies']]
                
                movie_details = {
                    'director': ', '.join(directors),
                    'cast': ', '.join(cast),
                    'tags': ', '.join(tags),
                    'runtime': data.get('runtime'),
                    'budget': data.get('budget'),
                    'revenue': data.get('revenue'),
                    'original_language': data.get('original_language'),
                    'production_companies': ', '.join(production_companies),
                    'similar_movies': [movie['id'] for movie in data['recommendations']['results'][:5]]
                }
                
                return movie_details
                
        except Exception as e:
            logging.error(f"Error fetching details for movie {movie_id}: {str(e)}")
            return None
    
    def get_popular_movies(self, num_pages=3):
        """Fetch popular movies with enhanced details from TMDB"""
        print("🎬 Fetching popular movies with enhanced details...")
        
        for page in range(1, num_pages + 1):
            url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                'api_key': self.tmdb_api_key,
                'sort_by': 'popularity.desc',
                'page': page,
                'vote_count.gte': 100
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    movies = response.json()['results']
                    
                    for movie in movies:
                        # Get additional details
                        details = self.get_full_movie_details(movie['id'])
                        
                        if details:
                            # Convert genre IDs to names
                            genres = [self.genre_mapping.get(genre_id, str(genre_id)) 
                                    for genre_id in movie['genre_ids']]
                            
                            movie_info = {
                                'movie_id': movie['id'],
                                'title': movie['title'],
                                'director': details['director'],
                                'cast': details['cast'],
                                'release_date': movie['release_date'],
                                'description': movie['overview'],
                                'categories': ', '.join(genres),
                                'tags': details['tags'],
                                'rating': movie['vote_average'],
                                'votes': movie['vote_count'],
                                'popularity': movie['popularity'],
                                'runtime': details['runtime'],
                                'budget': details['budget'],
                                'revenue': details['revenue'],
                                'original_language': details['original_language'],
                                'production_companies': details['production_companies'],
                                'similar_movie_ids': details['similar_movies']
                            }
                            self.movies_data.append(movie_info)
                            
                            print(f"📥 Collected '{movie['title']}' ({page}/{num_pages})")
                            time.sleep(1)  # TMDB rate limit. Adds a 1-second delay between TMDB API requests.
                
            except Exception as e:
                logging.error(f"Error fetching movies on page {page}: {str(e)}")
                print(f"❌ Error on page {page}: {str(e)}")
                
    def clean_title_for_url(self, title):
        """Clean movie title for Letterboxd URL format"""
        
        # Parse the title for Letterboxd URL format
        # We need this since Letterboxd has specific URL patterns
        # Remove special characters and convert to lowercase
        clean = re.sub(r'[^\w\s-]', '', title.lower())
        # Replace spaces with hyphens and remove multiple hyphens
        clean = re.sub(r'[-\s]+', '-', clean).strip('-')
        return clean
    
    def verify_letterboxd_url(self, url):
        """Try to verify if a Letterboxd URL exists"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.head(url, headers=headers, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
        
        
    def get_letterboxd_reviews(self, max_movies=None):
        """Scrape user reviews from Letterboxd"""
        print("\n🎯 Getting user reviews from Letterboxd...")
        
        # Use all movies or just a subset
        movies_to_scrape = self.movies_data[:max_movies] if max_movies else self.movies_data
        
        for i, movie in enumerate(movies_to_scrape, 1):
            # Format movie title for URL
            movie_title = self.clean_title_for_url(movie['title'])
            movie_year = movie['release_date'].split('-')[0]
            
            
            #url = f"https://letterboxd.com/film/{movie_title}-{movie_year}/reviews/by/activity/"
            
            # Try different URL patterns since Letterboxd isn't consistent
            url_patterns = [
                f"https://letterboxd.com/film/{movie_title}/reviews/by/activity/",
                f"https://letterboxd.com/film/{movie_title}-{movie_year}/reviews/by/activity/",
            ]
            
            valid_url = None
            for url in url_patterns:
                if self.verify_letterboxd_url(url):
                    valid_url = url
                    break
            
            if not valid_url:
                print(f"⚠️ Could not find Letterboxd page for {movie['title']}")
                continue
            
            # Use a regular browser header
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                response = requests.get(valid_url, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find all review containers
                    review_cards = soup.find_all('div', {'class': ['review-tile', 'film-detail-content']})
                    
                    if not review_cards:
                        print(f"⚠️ No reviews found for {movie['title']}")
                        continue
                    
                    reviews_added = 0
                    for card in review_cards[:5]:  # Get first 5 reviews per movie
                        # Get review text
                        review_text = card.find('div', class_='body-text')
                        review_text = review_text.get_text().strip() if review_text else ""
                        
                        # Get rating if it exists
                        rating_elem = card.find('span', class_='rating')
                        rating = None
                        if rating_elem and 'rated-' in rating_elem.get('class', [''])[1]:
                            rating = int(rating_elem.get('class', [''])[1][-1])
                        
                        # username extraction
                        username_elem = card.find('strong', {'class': 'name'})
                        if username_elem:
                            # Get the username text and clean it
                            username = username_elem.get_text().strip()
                            # Remove any emojis or special characters, keep only alphanumeric and some special chars
                            username = re.sub(r'[^\w\s@._-]', '', username)
                            # Replace spaces with underscores and convert to lowercase for consistency
                            username = username.replace(' ', '_').lower()
                        else:
                            print(f"Debug: Could not find username for review. HTML structure: {card.prettify()}")
                            username = "anonymous"
                            
                        
                        review_info = {
                            'movie_id': movie['movie_id'],
                            'movie_title': movie['title'],
                            'username': username,
                            'rating': rating,
                            'review': review_text,
                            'scrape_date': datetime.now().strftime('%Y-%m-%d')
                        }
                        self.reviews_data.append(review_info)
                        reviews_added += 1
                        
                        if username == "anonymous":
                            print(f"Warning: Anonymous username found for review of {movie['title']}")
                    
                    print(f"📝 Got reviews for {movie['title']} ({i}/{len(movies_to_scrape)})")
                    time.sleep(2)  # Letterboxd rate limit. Adds a 2-second delay between requests to Letterboxd
                    
            except Exception as e:
                logging.error(f"Error getting reviews for {movie['title']}: {str(e)}")
                print(f"❌ Couldn't get reviews for {movie['title']}")

    def save_data(self):
        """Save collected data with additional processing"""
        print("\n💾 Saving enhanced dataset...")
        
        # Convert to DataFrames
        movies_df = pd.DataFrame(self.movies_data)
        reviews_df = pd.DataFrame(self.reviews_data)
        
        # Additional data cleaning
        movies_df['release_year'] = pd.to_datetime(movies_df['release_date']).dt.year
        
        # Create additional features for the recommendation system
        movies_df['has_high_budget'] = movies_df['budget'] > movies_df['budget'].median()
        movies_df['is_successful'] = movies_df['revenue'] > movies_df['budget']
        
        # Save processed data
        movies_df.to_csv('data/movies.csv', index=False)
        reviews_df.to_csv('data/reviews.csv', index=False)
        
        # Save metadata
        metadata = {
            'total_movies': len(movies_df),
            'total_reviews': len(reviews_df),
            'unique_directors': movies_df['director'].nunique(),
            'unique_cast_members': len(set(','.join(movies_df['cast'].fillna('')).split(','))),
            'genres': list(set(','.join(movies_df['categories'].fillna('')).split(','))),
            'avg_rating': movies_df['rating'].mean(),
            'scrape_date': datetime.now().strftime('%Y-%m-%d'),
        }
        
        with open('data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Successfully saved {len(movies_df)} movies and {len(reviews_df)} reviews")
        print(f"📊 Dataset includes {metadata['unique_directors']} unique directors and {metadata['unique_cast_members']} cast members")

def main():
    try:
        scraper = EnhancedMovieScraper()
        
        # Get 10 pages of popular movies with enhanced details
        scraper.get_popular_movies(num_pages=10)
        
        # Get reviews for up to 200 movies
        scraper.get_letterboxd_reviews(max_movies=300)
        
        # Save everything
        scraper.save_data()
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        print(f"❌ Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()