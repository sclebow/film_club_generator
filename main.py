import streamlit as st
import pandas as pd
import plotly.express as px
import gzip
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from stqdm import stqdm

# IMDb dataset URLs
TITLE_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
TITLE_CREW_URL = "https://datasets.imdbws.com/title.crew.tsv.gz"
NAME_BASICS_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"
TITLE_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
TITLE_PRINCIPALS_URL = "https://datasets.imdbws.com/title.principals.tsv.gz"

# Pre-filtered data file
FILTERED_DATA_FILE = "imdb_movies_directors_actors.pkl"

def prepare_imdb_file(url, filename):
    """Download or reuse a fresh IMDb dataset on the Streamlit thread."""
    filepath = Path(filename)
    print(f"[IMDb] Preparing {filename}", flush=True)
    
    # Check if file exists and is older than 7 days
    if filepath.exists():
        file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        if file_age > timedelta(days=7):
            st.info(f"{filename} is older than 7 days. Deleting and re-downloading...")
            print(f"[IMDb] {filename} is stale; downloading a fresh copy", flush=True)
            filepath.unlink()
        else:
            st.info(f"{filename} already exists and is up to date.")
            print(f"[IMDb] Using existing {filename} ({filepath.stat().st_size:,} bytes)", flush=True)
    
    if not filepath.exists():
        st.info(f"Downloading {filename}...")
        print(f"[IMDb] Downloading {filename} from {url}", flush=True)
        tmp_path = filepath.with_name(filepath.name + ".part")
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        block_size = 1024 * 1024
        with open(tmp_path, 'wb') as f:
            with stqdm(total=total_size, desc=f"Downloading {filename}", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))
        tmp_path.rename(filepath)
        print(f"[IMDb] Downloaded {filename} ({filepath.stat().st_size:,} bytes)", flush=True)
        
        st.success(f"Downloaded {filename}")
    return filepath


def read_imdb_file(filepath, usecols=None, chunk_filter=None, chunksize=1_000_000):
    """Read an IMDb dataset in chunks, optionally filtering each chunk."""
    filename = filepath.name
    print(f"[IMDb] Reading {filename} in chunks of {chunksize:,} rows", flush=True)
    with pd.read_csv(
        filepath, sep='\t', na_values='\\N', usecols=usecols,
        chunksize=chunksize, compression='gzip', quoting=3,
        dtype=str, encoding='utf-8'
    ) as reader:
        parts = []
        rows_read = 0
        rows_kept = 0
        for chunk_number, chunk in enumerate(reader, start=1):
            rows_read += len(chunk)
            print(f"[IMDb] {filename}: read chunk {chunk_number} ({rows_read:,} rows total); filtering", flush=True)
            if chunk_filter is not None:
                chunk = chunk_filter(chunk)
            rows_kept += len(chunk)
            print(f"[IMDb] {filename}: finished chunk {chunk_number} ({rows_kept:,} rows kept)", flush=True)
            parts.append(chunk)
    print(f"[IMDb] {filename}: combining {len(parts):,} chunks ({rows_kept:,} rows)", flush=True)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
    print(f"[IMDb] {filename}: dataframe ready ({len(df):,} rows)", flush=True)
    return df


def download_and_load_imdb_data(url, filename, usecols=None, chunk_filter=None, chunksize=1_000_000):
    """Download and load an IMDb dataset on the Streamlit thread."""
    filepath = prepare_imdb_file(url, filename)
    with st.spinner(f"Loading {filename}..."):
        return read_imdb_file(filepath, usecols, chunk_filter, chunksize)

def create_filtered_dataset(parallel_load=True):
    """Download raw data and create pre-filtered pickle file"""
    st.info("Creating pre-filtered dataset from IMDb data. This will take several minutes...")
    print("[IMDb] Starting filtered dataset build", flush=True)
    
    current_year = datetime.now().year

    def released_movies(chunk):
        years = pd.to_numeric(chunk['startYear'], errors='coerce')
        return chunk[chunk['titleType'].eq('movie') & years.le(current_year)]

    # Load datasets with only needed columns/rows (chunked to limit memory)
    title_basics = download_and_load_imdb_data(
        TITLE_BASICS_URL, "title.basics.tsv.gz",
        usecols=['tconst', 'titleType', 'primaryTitle', 'startYear', 'genres'],
        chunk_filter=released_movies
    )
    print(f"[IMDb] Building movie ID index from {len(title_basics):,} released movies", flush=True)
    movie_ids = pd.Index(title_basics['tconst'].dropna().unique())
    title_ratings = download_and_load_imdb_data(
        TITLE_RATINGS_URL, "title.ratings.tsv.gz",
        usecols=['tconst', 'averageRating', 'numVotes'],
        chunk_filter=lambda c: c[movie_ids.get_indexer(c['tconst']) >= 0]
    )
    title_ratings['averageRating'] = pd.to_numeric(title_ratings['averageRating'], errors='coerce')
    title_ratings['numVotes'] = pd.to_numeric(title_ratings['numVotes'], errors='coerce')

    crew_filter = lambda c: c[movie_ids.get_indexer(c['tconst']) >= 0]
    principals_filter = lambda c: c[
        (movie_ids.get_indexer(c['tconst']) >= 0) &
        c['category'].isin(['actor', 'actress', 'producer'])
    ]
    if parallel_load:
        crew_path = prepare_imdb_file(TITLE_CREW_URL, "title.crew.tsv.gz")
        principals_path = prepare_imdb_file(TITLE_PRINCIPALS_URL, "title.principals.tsv.gz")
        # Initialize the shared read-only lookup index before starting the workers.
        movie_ids.get_indexer(movie_ids[:1])
        print("[IMDb] Loading crew and principals in parallel (two threads)", flush=True)
        with st.spinner("Loading crew and principal credits in parallel..."):
            with ThreadPoolExecutor(max_workers=2) as executor:
                crew_future = executor.submit(
                    read_imdb_file, crew_path, ['tconst', 'directors'], crew_filter, 500_000
                )
                principals_future = executor.submit(
                    read_imdb_file, principals_path, ['tconst', 'nconst', 'category'], principals_filter, 500_000
                )
                title_crew = crew_future.result()
                title_principals = principals_future.result()
    else:
        title_crew = download_and_load_imdb_data(
            TITLE_CREW_URL, "title.crew.tsv.gz",
            usecols=['tconst', 'directors'], chunk_filter=crew_filter
        )
        title_principals = download_and_load_imdb_data(
            TITLE_PRINCIPALS_URL, "title.principals.tsv.gz",
            usecols=['tconst', 'nconst', 'category'], chunk_filter=principals_filter
        )
    del movie_ids
    
    # Filter for movies only and select needed columns
    print("[IMDb] Preparing and filtering movie dataframe", flush=True)
    movies = title_basics[['tconst', 'primaryTitle', 'startYear', 'genres']].copy()
    del title_basics

    movies['startYear'] = pd.to_numeric(movies['startYear'], errors='coerce')
    print(f"[IMDb] Kept {len(movies):,} released movies", flush=True)

    # Merge with ratings
    print(f"[IMDb] Merging {len(movies):,} movies with {len(title_ratings):,} ratings", flush=True)
    movies = movies.merge(
        title_ratings[['tconst', 'averageRating', 'numVotes']],
        on='tconst',
        how='left'
    )
    del title_ratings
    
    # Merge with crew data
    print(f"[IMDb] Merging movies with {len(title_crew):,} crew rows", flush=True)
    movies_crew = movies.merge(
        title_crew[['tconst', 'directors']], 
        on='tconst', 
        how='inner'
    )
    del title_crew
    
    # Remove entries without directors
    movies_crew = movies_crew[movies_crew['directors'].notna()].copy()
    
    # Split directors efficiently
    print(f"[IMDb] Splitting directors for {len(movies_crew):,} movies", flush=True)
    movies_exploded = movies_crew.assign(
        director_id=movies_crew['directors'].str.split(',')
    ).explode('director_id')
    del movies_crew
    
    # Remove whitespace from director IDs
    movies_exploded['director_id'] = movies_exploded['director_id'].str.strip()
    
    # Count movies per director and calculate popularity metrics
    print(f"[IMDb] Calculating director statistics from {len(movies_exploded):,} rows", flush=True)
    director_stats = movies_exploded.groupby('director_id').agg({
        'tconst': 'count',
        'averageRating': 'mean',
        'numVotes': 'sum'
    }).reset_index()
    director_stats.columns = ['director_id', 'movie_count', 'avg_rating', 'total_votes']
    
    # Calculate popularity score (weighted by votes and rating)
    director_stats['popularity_score'] = (
        director_stats['avg_rating'].fillna(0) * 
        (director_stats['total_votes'].fillna(0) / 1000).clip(upper=100)
    )
    
    # Get director info - only for directors who actually directed movies
    director_ids = set(director_stats['director_id'])
    print(f"[IMDb] Calculated statistics for {len(director_ids):,} directors", flush=True)
    
    # Process actors from title.principals
    # Filter for actors/actresses in movies
    print(f"[IMDb] Filtering {len(title_principals):,} principal rows for released movies", flush=True)
    actors_in_movies = title_principals[
        title_principals['category'].isin(['actor', 'actress'])
    ][['tconst', 'nconst']].copy()
    
    # Merge with movie data to get ratings and other info
    print(f"[IMDb] Merging {len(actors_in_movies):,} actor credits with movies", flush=True)
    actors_movies = actors_in_movies.merge(
        movies[['tconst', 'primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']],
        on='tconst',
        how='inner'
    )
    del actors_in_movies
    actors_movies.rename(columns={'nconst': 'actor_id'}, inplace=True)
    
    # Count movies per actor and calculate popularity metrics
    print(f"[IMDb] Calculating actor statistics from {len(actors_movies):,} rows", flush=True)
    actor_stats = actors_movies.groupby('actor_id').agg({
        'tconst': 'count',
        'averageRating': 'mean',
        'numVotes': 'sum'
    }).reset_index()
    actor_stats.columns = ['actor_id', 'movie_count', 'avg_rating', 'total_votes']
    
    # Calculate popularity score (weighted by votes and rating)
    actor_stats['popularity_score'] = (
        actor_stats['avg_rating'].fillna(0) * 
        (actor_stats['total_votes'].fillna(0) / 1000).clip(upper=100)
    )
    
    # Get actor info - only for actors who actually acted in movies
    actor_ids = set(actor_stats['actor_id'])
    print(f"[IMDb] Calculated statistics for {len(actor_ids):,} actors", flush=True)

    print("[IMDb] Preparing producer credits for released movies", flush=True)
    producers_in_movies = title_principals[
        title_principals['category'] == 'producer'
    ][['tconst', 'nconst']].drop_duplicates()
    print(f"[IMDb] Merging {len(producers_in_movies):,} producer credits with movies", flush=True)
    producers_movies = producers_in_movies.merge(
        movies[['tconst', 'primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']],
        on='tconst',
        how='inner'
    ).rename(columns={'nconst': 'producer_id'})
    del producers_in_movies, title_principals, movies
    print(f"[IMDb] Calculating producer statistics from {len(producers_movies):,} rows", flush=True)
    producer_stats = producers_movies.groupby('producer_id').agg({
        'tconst': 'count',
        'averageRating': 'mean',
        'numVotes': 'sum'
    }).reset_index()
    producer_stats.columns = ['producer_id', 'movie_count', 'avg_rating', 'total_votes']
    producer_stats['popularity_score'] = (
        producer_stats['avg_rating'].fillna(0) *
        (producer_stats['total_votes'].fillna(0) / 1000).clip(upper=100)
    )
    producer_ids = set(producer_stats['producer_id'])
    print(f"[IMDb] Calculated statistics for {len(producer_ids):,} producers", flush=True)

    needed_ids = director_ids | actor_ids | producer_ids
    needed_index = pd.Index(list(needed_ids))
    del needed_ids
    name_basics = download_and_load_imdb_data(
        NAME_BASICS_URL, "name.basics.tsv.gz",
        usecols=['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession'],
        chunk_filter=lambda c: c[needed_index.get_indexer(c['nconst']) >= 0]
    )
    del needed_index
    for col in ['birthYear', 'deathYear']:
        name_basics[col] = pd.to_numeric(name_basics[col], errors='coerce')
    print(f"[IMDb] Matching {len(name_basics):,} names to directors, actors and producers", flush=True)
    directors_info = name_basics[name_basics['nconst'].isin(director_ids)].copy()
    actors_info = name_basics[name_basics['nconst'].isin(actor_ids)].copy()
    producers_info = name_basics[name_basics['nconst'].isin(producer_ids)].copy()
    del name_basics
    
    # Save to pickle
    print(f"[IMDb] Saving filtered dataset to {FILTERED_DATA_FILE}", flush=True)
    with open(FILTERED_DATA_FILE, 'wb') as f:
        pickle.dump({
            'movies_exploded': movies_exploded,
            'director_stats': director_stats,
            'directors_info': directors_info,
            'actors_movies': actors_movies,
            'actor_stats': actor_stats,
            'actors_info': actors_info,
            'producers_movies': producers_movies,
            'producer_stats': producer_stats,
            'producers_info': producers_info,
            'created_at': datetime.now()
        }, f)
    
    print(f"[IMDb] Filtered dataset saved to {FILTERED_DATA_FILE}", flush=True)
    st.success(f"Pre-filtered dataset created: {FILTERED_DATA_FILE}")
    
    return (movies_exploded, director_stats, directors_info, actors_movies, actor_stats, actors_info,
            producers_movies, producer_stats, producers_info)

@st.cache_data(ttl=timedelta(days=7))
def load_and_process_base_data():
    """Load pre-filtered data or create it if needed"""
    filepath = Path(FILTERED_DATA_FILE)
    
    # Check if filtered file exists and is fresh
    if filepath.exists():
        file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        if file_age <= timedelta(days=7):
            with st.spinner("Loading pre-filtered dataset..."):
                print(f"[IMDb] Loading pre-filtered dataset from {filepath}", flush=True)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"[IMDb] Loaded pre-filtered dataset from {filepath}", flush=True)
                
                required_keys = {
                    'movies_exploded', 'director_stats', 'directors_info',
                    'actors_movies', 'actor_stats', 'actors_info',
                    'producers_movies', 'producer_stats', 'producers_info',
                }
                if not required_keys.issubset(data):
                    st.info("Old data format detected. Re-creating dataset with producers...")
                    print("[IMDb] Old dataset format detected; rebuilding", flush=True)
                    filepath.unlink()
                    return create_filtered_dataset(parallel_load=os.environ.get('FILM_CLUB_PARALLEL_LOAD', '1') == '1')
                
                st.success(f"Loaded pre-filtered data (age: {file_age.days} days)")
                return (data['movies_exploded'], data['director_stats'], data['directors_info'],
                        data['actors_movies'], data['actor_stats'], data['actors_info'],
                        data['producers_movies'], data['producer_stats'], data['producers_info'])
        else:
            st.info("Pre-filtered dataset is older than 7 days. Refreshing...")
            print("[IMDb] Pre-filtered dataset is stale; rebuilding", flush=True)
            filepath.unlink()
    
    # Create filtered dataset
    return create_filtered_dataset(parallel_load=os.environ.get('FILM_CLUB_PARALLEL_LOAD', '1') == '1')

def find_directors_with_n_movies(n_movies=12, max_movies=None):
    """Find directors within an inclusive movie-count range."""
    if max_movies is None:
        max_movies = n_movies
    
    # Load pre-processed data (cached)
    movies_exploded, director_stats, directors_info, *_ = load_and_process_base_data()
    
    directors_with_n = director_stats[director_stats['movie_count'].between(n_movies, max_movies)].copy()
    
    # Merge with names
    directors_with_names = directors_with_n.merge(
        directors_info,
        left_on='director_id',
        right_on='nconst',
        how='left'
    )
    
    # Sort by popularity score (highest first)
    directors_with_names = directors_with_names.sort_values('popularity_score', ascending=False)
    
    # Get movies only for these specific directors
    directors_movies = movies_exploded[
        movies_exploded['director_id'].isin(directors_with_n['director_id'])
    ][['director_id', 'primaryTitle', 'startYear', 'genres', 'tconst', 'averageRating', 'numVotes']].copy()
    
    return directors_with_names, directors_movies, director_stats

def find_actors_with_n_movies(n_movies=12, max_movies=None):
    """Find actors within an inclusive movie-count range."""
    if max_movies is None:
        max_movies = n_movies
    
    # Load pre-processed data (cached)
    _, _, _, actors_movies, actor_stats, actors_info, *_ = load_and_process_base_data()
    
    actors_with_n = actor_stats[actor_stats['movie_count'].between(n_movies, max_movies)].copy()
    
    # Merge with names
    actors_with_names = actors_with_n.merge(
        actors_info,
        left_on='actor_id',
        right_on='nconst',
        how='left'
    )
    
    # Sort by popularity score (highest first)
    actors_with_names = actors_with_names.sort_values('popularity_score', ascending=False)
    
    # Get movies only for these specific actors
    actors_in_movies = actors_movies[
        actors_movies['actor_id'].isin(actors_with_n['actor_id'])
    ][['actor_id', 'primaryTitle', 'startYear', 'genres', 'tconst', 'averageRating', 'numVotes']].copy()
    
    return actors_with_names, actors_in_movies, actor_stats

def find_producers_with_n_movies(n_movies=12, max_movies=None):
    """Find producers within an inclusive movie-count range."""
    if max_movies is None:
        max_movies = n_movies
    _, _, _, _, _, _, producers_movies, producer_stats, producers_info = load_and_process_base_data()
    producers_with_n = producer_stats[producer_stats['movie_count'].between(n_movies, max_movies)].copy()
    producers_with_names = producers_with_n.merge(
        producers_info, left_on='producer_id', right_on='nconst', how='left'
    ).sort_values('popularity_score', ascending=False)
    movies_for_producers = producers_movies[
        producers_movies['producer_id'].isin(producers_with_n['producer_id'])
    ][['producer_id', 'primaryTitle', 'startYear', 'genres', 'tconst', 'averageRating', 'numVotes']].copy()
    return producers_with_names, movies_for_producers, producer_stats

def movie_count_label(movie_range):
    minimum, maximum = movie_range
    return f"{minimum}" if minimum == maximum else f"{minimum}–{maximum}"


def highlight_movie_range(fig, movie_range):
    minimum, maximum = movie_range
    if minimum == maximum:
        fig.add_vline(x=minimum, line_dash="dash", line_color="red",
                      annotation_text=f"{minimum} movies")
    else:
        fig.add_vrect(x0=minimum - 0.5, x1=maximum + 0.5,
                      fillcolor="red", opacity=0.15, line_width=0)


def main():
    st.set_page_config(page_title="IMDb Film People Analysis", page_icon="🎬", layout="wide")
    
    st.title("🎬 IMDb Directors, Actors & Producers Analysis")
    st.markdown("### Find Directors, Actors and Producers by Movie Count")
    
    st.info("""
    This app analyzes the official IMDb datasets to find directors, actors and producers within a movie-count range.
    Actor and producer results are limited to credits included in IMDb's title.principals dataset.
    The first run will download the datasets (~2GB total), which may take several minutes.
    Subsequent runs will use cached data. Cache is refreshed every 7 days.
    """)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        options=["Directors", "Actors", "Both", "Producers", "All"],
        index=2,
        help="Both analyzes directors and actors; All includes producers too"
    )
    
    movie_range = st.sidebar.slider(
        "Number of movies",
        min_value=1,
        max_value=50,
        value=(12, 12),
        help="Select an inclusive minimum and maximum number of movies"
    )
    count_label = movie_count_label(movie_range)
    
    try:
        # Load data based on analysis type
        if analysis_type in ["Directors", "Both", "All"]:
            directors_df, directors_movies_df, director_stats = find_directors_with_n_movies(*movie_range)
        
        if analysis_type in ["Actors", "Both", "All"]:
            actors_df, actors_movies_df, actor_stats = find_actors_with_n_movies(*movie_range)
        if analysis_type in ["Producers", "All"]:
            producers_df, producers_movies_df, producer_stats = find_producers_with_n_movies(*movie_range)
        
        # Display results summary
        if analysis_type == "Directors":
            st.success(f"Found {len(directors_df)} directors who have directed {count_label} movies")
        elif analysis_type == "Actors":
            st.success(f"Found {len(actors_df)} actors who have appeared in {count_label} movies")
        elif analysis_type == "Producers":
            st.success(f"Found {len(producers_df)} producers credited on {count_label} movies")
        elif analysis_type == "Both":
            st.success(f"Found {len(directors_df)} directors and {len(actors_df)} actors with {count_label} movies")
        else:
            st.success(f"Found {len(directors_df)} directors, {len(actors_df)} actors and {len(producers_df)} producers with {count_label} movies")
        
        # Create main sections based on analysis type
        if analysis_type == "Directors":
            create_director_tabs(movie_range, directors_df, directors_movies_df, director_stats)
        elif analysis_type == "Actors":
            create_actor_tabs(movie_range, actors_df, actors_movies_df, actor_stats)
        elif analysis_type == "Producers":
            create_credit_tabs(movie_range, producers_df, producers_movies_df, producer_stats, "producer")
        elif analysis_type == "Both":
            main_tab1, main_tab2 = st.tabs(["🎬 Directors", "🎭 Actors"])
            
            with main_tab1:
                create_director_tabs(movie_range, directors_df, directors_movies_df, director_stats)
            
            with main_tab2:
                create_actor_tabs(movie_range, actors_df, actors_movies_df, actor_stats)
        else:
            main_tab1, main_tab2, main_tab3 = st.tabs(["🎬 Directors", "🎭 Actors", "🎞️ Producers"])
            with main_tab1:
                create_director_tabs(movie_range, directors_df, directors_movies_df, director_stats)
            with main_tab2:
                create_actor_tabs(movie_range, actors_df, actors_movies_df, actor_stats)
            with main_tab3:
                create_credit_tabs(movie_range, producers_df, producers_movies_df, producer_stats, "producer")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

def create_director_tabs(movie_range, directors_df, movies_df, all_counts):
    """Create tabs for director analysis"""
    count_label = movie_count_label(movie_range)
    filename_count = count_label.replace("–", "-")
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Directors List", "🎥 Movies by Director", "📊 Distribution", "📈 Statistics"])
    
    with tab1:
        st.subheader(f"Directors with {count_label} Movies (Sorted by Popularity)")
        
        display_df = directors_df[['primaryName', 'avg_rating', 'total_votes', 'birthYear', 'deathYear', 'primaryProfession', 'movie_count']].copy()
        display_df.columns = ['Name', 'Avg Rating', 'Total Votes', 'Birth Year', 'Death Year', 'Professions', 'Movie Count']
        display_df['Avg Rating'] = display_df['Avg Rating'].round(1)
        
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
            file_name=f"directors_with_{filename_count}_movies.csv",
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
            director_movies = director_movies.sort_values('startYear')
            
            st.write(f"**{selected_director}** directed these {len(director_movies)} movies:")
            
            movie_display = director_movies[['primaryTitle', 'startYear', 'averageRating', 'numVotes', 'genres']].copy()
            movie_display.columns = ['Title', 'Year', 'Rating', 'Votes', 'Genres']
            movie_display['Rating'] = movie_display['Rating'].round(1)
            movie_display = movie_display.sort_values('Rating', ascending=False)
            
            st.dataframe(movie_display, use_container_width=True, height=465)
    
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
        
        highlight_movie_range(fig, movie_range)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Overall Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Directors in Dataset", f"{len(all_counts):,}")
        
        with col2:
            st.metric(f"Directors with {count_label} Movies", f"{len(directors_df):,}")
        
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

def create_actor_tabs(movie_range, actors_df, movies_df, all_counts):
    """Create tabs for actor analysis"""
    create_credit_tabs(movie_range, actors_df, movies_df, all_counts, "actor")

def create_credit_tabs(movie_range, actors_df, movies_df, all_counts, role):
    """Create tabs for analysis of principal credits."""
    count_label = movie_count_label(movie_range)
    filename_count = count_label.replace("–", "-")
    singular = role.capitalize()
    plural = f"{singular}s"
    id_column = f"{role}_id"
    tab1, tab2, tab3, tab4 = st.tabs([f"📋 {plural} List", f"🎥 Movies by {singular}", "📊 Distribution", "📈 Statistics"])
    
    with tab1:
        st.subheader(f"{plural} with {count_label} Movies (Sorted by Popularity)")
        
        display_df = actors_df[['primaryName', 'avg_rating', 'total_votes', 'birthYear', 'deathYear', 'primaryProfession', 'movie_count']].copy()
        display_df.columns = ['Name', 'Avg Rating', 'Total Votes', 'Birth Year', 'Death Year', 'Professions', 'Movie Count']
        display_df['Avg Rating'] = display_df['Avg Rating'].round(1)
        
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
            file_name=f"{role}s_with_{filename_count}_movies.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader(f"Browse Movies by {singular}")
        
        if len(actors_df) > 0:
            selected_actor = st.selectbox(
                f"Select {'an' if role == 'actor' else 'a'} {role}",
                options=actors_df['primaryName'].tolist(),
                index=0
            )
            
            actor_id = actors_df[actors_df['primaryName'] == selected_actor][id_column].iloc[0]
            actor_movies = movies_df[movies_df[id_column] == actor_id].copy()
            actor_movies = actor_movies.sort_values('startYear')
            
            verb = "appeared in" if role == "actor" else "was credited as a producer on"
            st.write(f"**{selected_actor}** {verb} these {len(actor_movies)} movies:")
            
            movie_display = actor_movies[['primaryTitle', 'startYear', 'averageRating', 'numVotes', 'genres']].copy()
            movie_display.columns = ['Title', 'Year', 'Rating', 'Votes', 'Genres']
            movie_display['Rating'] = movie_display['Rating'].round(1)
            movie_display = movie_display.sort_values('Rating', ascending=False)
            
            st.dataframe(movie_display, use_container_width=True, height=465)
    
    with tab3:
        st.subheader(f"Distribution of {singular} Movie Counts")
        
        # Show distribution of all movie counts
        count_dist = all_counts['movie_count'].value_counts().reset_index()
        count_dist.columns = ['Number of Movies', f'Number of {plural}']
        count_dist = count_dist.sort_values('Number of Movies')
        
        # Limit to reasonable range for visualization
        count_dist_filtered = count_dist[count_dist['Number of Movies'] <= 100]
        
        fig = px.bar(
            count_dist_filtered,
            x='Number of Movies',
            y=f'Number of {plural}',
            title=f'Distribution of Movie Counts per {singular} (up to 100 movies)',
            log_y=True
        )
        
        highlight_movie_range(fig, movie_range)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Overall Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Total {plural} in Dataset", f"{len(all_counts):,}")
        
        with col2:
            st.metric(f"{plural} with {count_label} Movies", f"{len(actors_df):,}")
        
        with col3:
            percentage = (len(actors_df) / len(all_counts) * 100) if len(all_counts) > 0 else 0
            st.metric("Percentage", f"{percentage:.2f}%")
        
        st.divider()
        
        # Show some interesting stats
        st.markdown("#### Movie Count Statistics")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write(f"**Most Prolific {plural}:**")
            data = load_and_process_base_data()
            people_info = data[5] if role == "actor" else data[8]
            top_actors = all_counts.nlargest(10, 'movie_count').merge(
                people_info[['nconst', 'primaryName']],
                left_on=id_column,
                right_on='nconst',
                how='left'
            )
            st.dataframe(
                top_actors[['primaryName', 'movie_count']].head(10),
                hide_index=True,
                use_container_width=True
            )
        
        with stats_col2:
            st.write("**Movie Count Distribution:**")
            st.write(f"- Mean: {all_counts['movie_count'].mean():.1f} movies")
            st.write(f"- Median: {all_counts['movie_count'].median():.0f} movies")
            st.write(f"- Max: {all_counts['movie_count'].max()} movies")
            st.write(f"- {plural} with 1 movie: {len(all_counts[all_counts['movie_count'] == 1]):,}")
            st.write(f"- {plural} with 10+ movies: {len(all_counts[all_counts['movie_count'] >= 10]):,}")

if __name__ == "__main__":
    main()