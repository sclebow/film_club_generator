import streamlit as st
import pandas as pd
import plotly.express as px
import gzip
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import os

# IMDb dataset URLs
TITLE_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
TITLE_CREW_URL = "https://datasets.imdbws.com/title.crew.tsv.gz"
NAME_BASICS_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"

@st.cache_data(ttl=timedelta(days=7))
def download_and_load_imdb_data(url, filename):
    """Download and load IMDb dataset"""
    filepath = Path(filename)
    
    # Check if file exists and is older than 7 days
    if filepath.exists():
        file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        if file_age > timedelta(days=7):
            st.info(f"{filename} is older than 7 days. Deleting and re-downloading...")
            filepath.unlink()
        else:
            st.info(f"{filename} already exists and is up to date.")
    
    if not filepath.exists():
        with st.spinner(f"Downloading {filename}..."):
            urllib.request.urlretrieve(url, filepath)
    
    with st.spinner(f"Loading {filename}..."):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, sep='\t', na_values='\\N', low_memory=False)
    
    return df

@st.cache_data(ttl=timedelta(days=7))
def find_directors_with_n_movies(n_movies=12):
    """Find directors who have directed exactly n movies"""
    
    # Load datasets
    title_basics = download_and_load_imdb_data(TITLE_BASICS_URL, "title.basics.tsv.gz")
    title_crew = download_and_load_imdb_data(TITLE_CREW_URL, "title.crew.tsv.gz")
    name_basics = download_and_load_imdb_data(NAME_BASICS_URL, "name.basics.tsv.gz")
    
    # Filter for movies only (not TV shows, episodes, etc.)
    movies = title_basics[title_basics['titleType'] == 'movie'].copy()
    
    # Merge with crew data to get directors
    movies_crew = movies.merge(title_crew[['tconst', 'directors']], on='tconst', how='inner')
    
    # Remove entries without directors
    movies_crew = movies_crew[movies_crew['directors'].notna()].copy()
    
    # Split directors (some movies have multiple directors)
    movies_exploded = movies_crew.assign(
        director_id=movies_crew['directors'].str.split(',')
    ).explode('director_id')
    
    # Count movies per director
    director_counts = movies_exploded.groupby('director_id').size().reset_index(name='movie_count')
    
    # Filter directors with exactly n movies
    directors_with_n = director_counts[director_counts['movie_count'] == n_movies]
    
    # Get director names
    directors_with_names = directors_with_n.merge(
        name_basics[['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession']],
        left_on='director_id',
        right_on='nconst',
        how='left'
    )
    
    # Get the movies for these directors
    directors_movies = movies_exploded[
        movies_exploded['director_id'].isin(directors_with_n['director_id'])
    ][['director_id', 'primaryTitle', 'startYear', 'genres', 'tconst']].copy()
    
    return directors_with_names, directors_movies, director_counts

def main():
    st.set_page_config(page_title="IMDb Directors Analysis", page_icon="🎬", layout="wide")
    
    st.title("🎬 IMDb Directors Analysis")
    st.markdown("### Find Directors Who Have Directed Exactly 12 Movies")
    
    st.info("""
    This app analyzes the official IMDb datasets to find directors who have directed exactly 12 movies.
    The first run will download the datasets (~1GB total), which may take a few minutes.
    Subsequent runs will use cached data.
    """)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    n_movies = st.sidebar.slider(
        "Number of movies",
        min_value=1,
        max_value=50,
        value=12,
        help="Find directors who directed exactly this many movies"
    )
    
    try:
        # Load and process data
        directors_df, movies_df, all_counts = find_directors_with_n_movies(n_movies)
        
        # Display results
        st.success(f"Found {len(directors_df)} directors who have directed exactly {n_movies} movies")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Directors List", "🎥 Movies by Director", "📊 Distribution", "📈 Statistics"])
        
        with tab1:
            st.subheader(f"Directors with Exactly {n_movies} Movies")
            
            display_df = directors_df[['primaryName', 'birthYear', 'deathYear', 'primaryProfession', 'movie_count']].copy()
            display_df.columns = ['Name', 'Birth Year', 'Death Year', 'Professions', 'Movie Count']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"directors_with_{n_movies}_movies.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("Browse Movies by Director")
            
            if len(directors_df) > 0:
                selected_director = st.selectbox(
                    "Select a director",
                    options=directors_df['primaryName'].tolist(),
                    index=0
                )
                
                director_id = directors_df[directors_df['primaryName'] == selected_director]['director_id'].iloc[0]
                director_movies = movies_df[movies_df['director_id'] == director_id].copy()
                
                st.write(f"**{selected_director}** directed these {len(director_movies)} movies:")
                
                movie_display = director_movies[['primaryTitle', 'startYear', 'genres']].copy()
                movie_display.columns = ['Title', 'Year', 'Genres']
                movie_display = movie_display.sort_values('Year')
                
                st.dataframe(movie_display, use_container_width=True)
        
        with tab3:
            st.subheader("Distribution of Director Movie Counts")
            
            # Show distribution of all movie counts
            count_dist = all_counts['movie_count'].value_counts().reset_index()
            count_dist.columns = ['Number of Movies', 'Number of Directors']
            count_dist = count_dist.sort_values('Number of Movies')
            
            # Limit to reasonable range for visualization
            count_dist_filtered = count_dist[count_dist['Number of Movies'] <= 100]
            
            fig = px.bar(
                count_dist_filtered,
                x='Number of Movies',
                y='Number of Directors',
                title='Distribution of Movie Counts per Director (up to 100 movies)',
                log_y=True
            )
            
            # Highlight the selected n_movies
            fig.add_vline(x=n_movies, line_dash="dash", line_color="red", 
                         annotation_text=f"{n_movies} movies")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Overall Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Directors in Dataset", f"{len(all_counts):,}")
            
            with col2:
                st.metric(f"Directors with {n_movies} Movies", f"{len(directors_df):,}")
            
            with col3:
                percentage = (len(directors_df) / len(all_counts) * 100) if len(all_counts) > 0 else 0
                st.metric("Percentage", f"{percentage:.2f}%")
            
            st.divider()
            
            # Show some interesting stats
            st.markdown("#### Movie Count Statistics")
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write("**Most Prolific Directors:**")
                top_directors = all_counts.nlargest(10, 'movie_count').merge(
                    directors_df[['director_id', 'primaryName']],
                    on='director_id',
                    how='left'
                )
                if 'primaryName' not in top_directors.columns:
                    # Load names for top directors if not already there
                    name_basics = download_and_load_imdb_data(NAME_BASICS_URL, "name.basics.tsv.gz")
                    top_directors = top_directors.merge(
                        name_basics[['nconst', 'primaryName']],
                        left_on='director_id',
                        right_on='nconst',
                        how='left'
                    )
                st.dataframe(
                    top_directors[['primaryName', 'movie_count']].head(10),
                    hide_index=True,
                    use_container_width=True
                )
            
            with stats_col2:
                st.write("**Movie Count Distribution:**")
                st.write(f"- Mean: {all_counts['movie_count'].mean():.1f} movies")
                st.write(f"- Median: {all_counts['movie_count'].median():.0f} movies")
                st.write(f"- Max: {all_counts['movie_count'].max()} movies")
                st.write(f"- Directors with 1 movie: {len(all_counts[all_counts['movie_count'] == 1]):,}")
                st.write(f"- Directors with 10+ movies: {len(all_counts[all_counts['movie_count'] >= 10]):,}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()