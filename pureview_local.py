import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import re
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import swifter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocesses a text string by lowercasing, removing punctuation, tokenizing, 
    removing stopwords, and lemmatizing.
    Parameters
    ----------
    text : str
        The input text to preprocess.
    Returns
    -------
    str
        The preprocessed text string, or an empty string if input is invalid.
    """
    if not isinstance(text, str):
        # Handle non-string inputs (e.g., NaN or numbers)
        return ""
    
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    return ' '.join(tokens)


def compute_tfidf_topics(df, text_column, top_n=3, max_features=5000):
    """
    Computes TF-IDF topics for the given text column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing preprocessed text.
    text_column : str
        The name of the column containing preprocessed text.
    top_n : int, optional
        The number of top words to extract for each row (default is 3).
    max_features : int, optional
        The maximum number of features for the TF-IDF vectorizer (default is 5000).

    Returns
    -------
    pandas.DataFrame
        The DataFrame with additional columns for the top N TF-IDF words.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    logging.info("Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    logging.info("Extracting top TF-IDF words...")
    top_words = tfidf_df.apply(lambda row: row.nlargest(top_n).index.tolist(), axis=1)
    topics_df = pd.DataFrame(top_words.tolist(), columns=[f"topic_{i+1}" for i in range(top_n)])
    return pd.concat([df, topics_df], axis=1)

def compute_sentiment(df, text_column, sentiment_column):
    """
    Computes VADER sentiment scores for a specified text column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing preprocessed text.
    text_column : str
        The name of the column containing preprocessed text.
    sentiment_column : str
        The name of the column to store sentiment scores.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with an additional column for sentiment scores.
    """
    sia = SentimentIntensityAnalyzer()
    logging.info("Computing sentiment scores...")
    df[sentiment_column] = df[text_column].swifter.apply(lambda text: sia.polarity_scores(text)['compound'])
    return df

def run_pureview_ai(file_path, num_rows=None, chunksize=10000):
    """
    Main function to process data for PureView AI pipeline.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file.
    num_rows : int, optional
        Number of rows to read from the file (default is None, meaning all rows).
    chunksize : int, optional
        Chunk size for batch processing (default is 10000).

    Returns
    -------
    pandas.DataFrame
        The final DataFrame with processed text, TF-IDF topics, and sentiment scores.
    """
    logging.info("Starting PureView AI pipeline...")
    
    if num_rows:
        logging.info(f"Reading the first {num_rows} rows from the file...")
        df = pd.read_csv(file_path, nrows=num_rows)
    else:
        logging.info("Reading the entire file in chunks...")
        df = pd.concat(
            pd.read_csv(file_path, chunksize=chunksize), 
            ignore_index=True
        )
    
    # Step 1: Preprocess the text
    logging.info("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Step 2: Drop rows with missing processed text
    df = df.dropna(subset=['processed_text'])
    
    # Step 3: Compute TF-IDF topics
    df = compute_tfidf_topics(df, text_column='processed_text')
    
    # Step 4: Compute sentiment scores
    df = compute_sentiment(df, text_column='processed_text', sentiment_column='sentiment_vader')
    
    logging.info("Pipeline completed.")
    return df

def output_pureview(df, bucket_name, destination_blob_name):
    """
    outputs the file to csv

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to upload.
    bucket_name : str
        The name of the GCS bucket.

    Returns
    -------
    None
    """
    logging.info("Uploading results to CSV...")

    local_path = './CSE-6242-Amazon-Review-Sentiment/data/pureview_ai.csv'
    df.to_csv(local_path, index=False)
    
if __name__ == "__main__":
    FILE_PATH = './CSE-6242-Amazon-Review-Sentiment/data/data.csv'
    BUCKET_NAME = 'amazon-home-and-kitchen'
    DESTINATION_BLOB_NAME = 'pureview_ai.csv'
    NUM_ROWS = 250000
    CHUNKSIZE = 10000
    
    final_df = run_pureview_ai(FILE_PATH, num_rows=NUM_ROWS, chunksize=CHUNKSIZE)
    output_pureview(final_df, bucket_name=BUCKET_NAME, destination_blob_name=DESTINATION_BLOB_NAME)
