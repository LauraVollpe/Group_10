import os
import tarfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pydantic import validate_arguments

class MovieAnalyzer:
    DATA_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DOWNLOAD_DIR = Path("downloads/")
    EXTRACT_DIR = DOWNLOAD_DIR / "MovieSummaries"

    def __init__(self):
        """Initialize the class: Download and load movie datasets."""
        self.DOWNLOAD_DIR.mkdir(exist_ok=True)

        # Paths
        self.data_path = self.DOWNLOAD_DIR / "MovieSummaries.tar.gz"
        
        # Download dataset if not already downloaded
        if not self.data_path.exists():
            print("Downloading dataset...")
            self._download_data()

        # Extract dataset if not already extracted
        if not self.EXTRACT_DIR.exists():
            print("Extracting dataset...")
            self._extract_data()

        # Load data into Pandas DataFrames
        self.movies_df = self._load_movies_data()
        self.actors_df = self._load_actors_data()

    def _download_data(self):
        """Download dataset if it does not exist."""
        response = requests.get(self.DATA_URL, stream=True)
        with open(self.data_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print("Download complete.")

    def _extract_data(self):
        """Extract dataset if it hasn't been extracted."""
        with tarfile.open(self.data_path, "r:gz") as tar:
            tar.extractall(self.EXTRACT_DIR)
        print("Extraction complete.")

    def _load_movies_data(self):
        """Load the movies dataset into a Pandas DataFrame."""
        movies_file = self.EXTRACT_DIR / "movie.metadata.tsv"
        if movies_file.exists():
            return pd.read_csv(movies_file, sep="\t", header=None)
        return pd.DataFrame()

    def _load_actors_data(self):
        """Load the actors dataset into a Pandas DataFrame."""
        actors_file = self.EXTRACT_DIR / "name.clusters"
        if actors_file.exists():
            return pd.read_csv(actors_file, sep="\t", header=None)
        return pd.DataFrame()

    @validate_arguments
    def movie_type(self, N: int = 10):
        """Return a DataFrame with the top N most common movie genres."""
        if 1 not in self.movies_df:
            raise ValueError("Movies dataset does not contain genre information.")
        
        genre_counts = self.movies_df[1].value_counts().head(N)
        return genre_counts.reset_index().rename(columns={"index": "Movie_Type", 1: "Count"})

    def actor_count(self):
        """Return a DataFrame showing how many movies have X number of actors."""
        if 0 not in self.actors_df:
            raise ValueError("Actors dataset does not contain movie IDs.")
        
        actor_counts = self.actors_df.groupby(0).size()
        histogram = actor_counts.value_counts().reset_index()
        histogram.columns = ["Number of Actors", "Movie Count"]
        return histogram

    @validate_arguments
    def actor_distributions(self, gender: str = "All", max_height: float = 200, min_height: float = 100, plot: bool = False):
        """
        Filter actors by gender and height range.
        If plot=True, displays a histogram of height distribution.
        """
        if not isinstance(gender, str):
            raise ValueError("Gender must be a string.")
        if not isinstance(max_height, (int, float)) or not isinstance(min_height, (int, float)):
            raise ValueError("Heights must be numerical values.")
        
        if 2 not in self.actors_df:
            raise ValueError("Actors dataset does not contain height information.")

        filtered_df = self.actors_df[(self.actors_df[2] >= min_height) & (self.actors_df[2] <= max_height)]

        if gender != "All":
            if 1 not in self.actors_df:
                raise ValueError("Actors dataset does not contain gender information.")
            filtered_df = filtered_df[filtered_df[1] == gender]

        if plot:
            plt.hist(filtered_df[2], bins=20, alpha=0.7)
            plt.xlabel("Height")
            plt.ylabel("Frequency")
            plt.title(f"Height Distribution for {gender}")
            plt.show()

        return filtered_df

