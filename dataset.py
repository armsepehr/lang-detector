import os
import pandas as pd
import requests
import bz2

# Constants
DATASET_URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
DATASET_BZ2_FILE = "sentences.tar.bz2"
DATASET_TSV_FILE = "sentences.csv"
LANGUAGES = {"eng": "English", "fra": "French", "spa": "Spanish", "deu": "German", "ita": "Italian"}

def download_dataset():
    """Downloads the dataset if it does not exist."""
    if not os.path.exists(DATASET_BZ2_FILE):
        print("Downloading dataset...")
        response = requests.get(DATASET_URL, stream=True)
        with open(DATASET_BZ2_FILE, "wb") as file:
            file.write(response.content)
        print("Download complete.")

def extract_dataset():
    """Extracts the dataset if not already extracted."""
    if not os.path.exists(DATASET_TSV_FILE):
        print("Extracting dataset...")
        with bz2.BZ2File(DATASET_BZ2_FILE, "rb") as file, open(DATASET_TSV_FILE, "wb") as out_file:
            out_file.write(file.read())
        print("Extraction complete.")

def load_dataset(sample_size=5000):
    """Loads and processes the dataset after ensuring it's available."""
    download_dataset()
    extract_dataset()
    
    print("Processing dataset...")
    
    # Load dataset
    df = pd.read_csv(DATASET_TSV_FILE, sep="\t", names=["id", "lang", "text"], usecols=["lang", "text"])

    # Filter selected languages
    df = df[df["lang"].isin(LANGUAGES.keys())]
    df["language"] = df["lang"].map(LANGUAGES)

    # Sample dataset to avoid memory issues
    df = df.groupby("language").apply(lambda x: x.sample(min(len(x), sample_size))).reset_index(drop=True)

    return df[["text", "language"]]

if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
