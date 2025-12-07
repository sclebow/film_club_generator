# IMDb Directors Analysis - Film Club Generator

A Streamlit application that analyzes the official IMDb datasets to find directors who have directed exactly 12 movies (or any specified number).

## Features

- 📊 **Directors List**: View all directors who have directed exactly N movies
- 🎥 **Movie Browser**: Browse movies by each director
- 📈 **Distribution Charts**: Visualize the distribution of movie counts across all directors
- 📊 **Statistics**: See overall statistics and most prolific directors
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
3. Process the data to find directors with exactly 12 movies (adjustable via slider)

## Data Source

This project uses the official IMDb datasets:
- `title.basics.tsv.gz` - Contains movie information
- `title.crew.tsv.gz` - Contains director information
- `name.basics.tsv.gz` - Contains people's names and details

Data is downloaded from: https://datasets.imdbws.com/

## Customization

Use the sidebar slider to change the number of movies from 1 to 50 to find directors who directed exactly that many films.

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly

## License

This project uses IMDb datasets which are available for personal and non-commercial use.
