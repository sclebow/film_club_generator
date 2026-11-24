# IMDb Film Club Generator

A Streamlit application that analyzes IMDb datasets to find directors, actors, and producers within a selected movie-count range.

## Features

- 📊 **People Lists**: View directors, actors, or producers within a movie-count range
- 🎥 **Movie Browser**: Browse movies by person
- 📈 **Distribution Charts**: Visualize movie counts for each role
- 📊 **Statistics**: See overall statistics and the most prolific people
- 💾 **Export Data**: Download results as CSV

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

The app will:
1. Download the IMDb datasets on first run (~1GB total)
2. Cache the data for faster subsequent loads
3. Process the data to find people with 12 movies by default (adjustable via slider)

Choose **Producers** for producer analysis or **All** to see all three roles.
**Both** retains the directors-and-actors view. Existing processed caches without
producer data are rebuilt on the next run. Producer results count distinct movies
with a `producer` credit in `title.principals`; this dataset contains principal
credits rather than every credit, so filmographies may be incomplete.
The initial build reads the compressed datasets in chunks and filters out
unreleased titles and unrelated records early. It can still take several minutes;
subsequent runs reuse the processed cache for up to seven days.

The build loads crew and principal credits using two parallel threads by default.
This may use more memory and can be slower on some machines. To load sequentially
when memory is tight, start the app with:

```bash
FILM_CLUB_PARALLEL_LOAD=0 streamlit run main.py
```

In PowerShell, use `$env:FILM_CLUB_PARALLEL_LOAD = "0"` before starting the app.
This only affects rebuilding the processed dataset, not loading an existing
cache. Files are still downloaded sequentially.

## Data Source

This project uses the official IMDb datasets:
- `title.basics.tsv.gz` - Contains movie information
- `title.crew.tsv.gz` - Contains director information
- `name.basics.tsv.gz` - Contains people's names and details
- `title.principals.tsv.gz` - Contains principal actor and producer credits
- `title.ratings.tsv.gz` - Contains ratings and vote counts

Data is downloaded from: https://datasets.imdbws.com/

## Customization

Use the sidebar to select the analysis type and an inclusive movie-count range
from 1 to 50. For example, 10–13 includes both 10 and 13. Set both slider
handles to the same value to search for an exact count.

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly

## License

This project uses IMDb datasets which are available for personal and non-commercial use.
