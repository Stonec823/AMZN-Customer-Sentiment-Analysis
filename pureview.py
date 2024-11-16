%%time

import pandas as pd
from google.cloud import storage
import gcsfs
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt') 
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

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
        The preprocessed text string.
    """
    if not isinstance(text, str):
        return text
    
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

def process_dataframe_in_batches(df, num_batches=5):
    """
    Processes a DataFrame in batches, applying text preprocessing to a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing text data to process.
    text_column : str
        The name of the column containing the raw text to preprocess.
    output_column : str
        The name of the column where the processed text will be stored.
    num_batches : int, optional
        The number of batches to split the DataFrame into for processing (default is 5).

    Returns
    -------
    pandas.DataFrame
        The concatenated DataFrame with the processed text in the specified output column.
    """
    # Split the DataFrame into batches
    df_batches = np.array_split(df, num_batches)
    processed_batches = []
    
    print('Pre processing now...')
    
    for i, batch in enumerate(df_batches):
        print(f"\n\tProcessing Batch {i + 1} of {num_batches}...")
        
        # Apply text preprocessing to the specified column
        batch['processed_text'] = batch['text'].apply(preprocess_text)
        processed_batches.append(batch)
    
    print('Pre processing complete...')
    # Concatenate processed batches into a single DataFrame
    return pd.concat(processed_batches, ignore_index=True)


def custom_tokenizer(text):
    """
    Tokenizes and filters a text string by removing numbers and unwanted words (e.g., stopwords).

    Parameters
    ----------
    text : str
        The input text to tokenize.

    Returns
    -------
    str
        A string of space-separated tokens containing only alphabetic words
        and excluding any stopwords.
    """
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())  # Only keep alphabetic words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


def get_top_words(row):
    """
    Extracts the top 3 words based on their TF-IDF scores from a DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A row of TF-IDF scores.

    Returns
    -------
    list of str
        A list of the top 3 words (or `None` for missing values if fewer than 3 words are available).
    """
    top_words = row.nlargest(3)  # Get the top 3 words based on TF-IDF score
    return top_words.index.tolist() if len(top_words) == 3 else [None, None, None]


def get_topics(df):
    """
    Processes a DataFrame to extract topics (top 3 words) from text using TF-IDF.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing a column named 'processed_text' with text data.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing the original data along with the filtered text and the top 3 topics.
        The returned DataFrame includes the following columns:
        - `review_id` : Original review ID.
        - `main_category` : Main category of the review.
        - `title_x` : Title of the review.
        - `rating` : Rating of the review.
        - `filtered_text` : Processed and filtered text.
        - `topic_1`, `topic_2`, `topic_3` : Top 3 topics extracted from the text.
    """
    dfc = df.copy()

    # Step 1: Preprocess `processed_text` by removing low-quality words
    dfc.loc[:, 'filtered_text'] = dfc['processed_text'].apply(custom_tokenizer)

    # Step 2: Define the TF-IDF vectorizer and fit it to the filtered text
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(dfc['filtered_text'])

    # Step 3: Create a DataFrame of the TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Step 5: Apply the function to each row to get the top 3 words
    dfc[['topic_1', 'topic_2', 'topic_3']] = tfidf_df.apply(get_top_words, axis=1, result_type="expand")

    return dfc

def sentiment_processing(df):
    """
    Processes a DataFrame to compute VADER sentiment scores for the 'filtered_text' column.

    This function splits the input DataFrame into batches, calculates the compound sentiment
    score for each text in the 'filtered_text' column using the VADER sentiment analyzer, 
    and returns a concatenated DataFrame with a new column 'sentiment_vader' containing the scores.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing a column named 'filtered_text', which is expected
        to have preprocessed text data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame identical to the input, but with an additional column:
        - `sentiment_vader`: The VADER compound sentiment score for each row.

    Notes
    -----
    - The DataFrame is processed in batches to handle large datasets efficiently.
    - The VADER sentiment analyzer computes a compound score, which ranges from -1 (most negative)
      to 1 (most positive).

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'filtered_text': ['I love this product!', 'This is terrible.', 'It is okay.']}
    >>> df = pd.DataFrame(data)
    >>> processed_df = sentiment_processing(df)
    >>> print(processed_df)
               filtered_text  sentiment_vader
    0  I love this product!           0.6369
    1     This is terrible.          -0.4767
    2          It is okay.           0.0000
    """

    num_batches = 100
    df_batches = np.array_split(df, num_batches)

    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    print('Sentiment processing now...')
    df_lst = []
    for i, batch in enumerate(df_batches):
        print(f"\n\tProcessing Batch {i + 1}...")
        batch['sentiment_vader'] = batch['filtered_text'].apply(lambda text: sia.polarity_scores(text)['compound'])
        df_lst.append(batch)
    print('Sentiment processing finished...')

    concatenated_df = pd.concat(df_lst, ignore_index=True)
    return concatenated_df

def run_pureview_ai(num_rows):
    fs = gcsfs.GCSFileSystem()
    path = f'gs://amazon-home-and-kitchen/full_train_data.csv'
    df = pd.read_csv(path, na_values=['—']
                     ,nrows=num_rows)
    
    df=process_dataframe_in_batches(df)
    df['review_id'] = df.index 
    df = df.dropna(subset=['processed_text'])

    df_topics = get_topics(df)
    final = sentiment_processing(df_topics)    
    return final

def output_pureview(df):
    bucket_name = 'amazon-home-and-kitchen'
    destination_blob_name = 'pureview_ai.csv'

    # Save DataFrame as CSV locally first
    final.to_csv('/tmp/pureview_ai.csv', index=False)

    # Initialize a GCS client and upload wwthe file
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename('/tmp/pureview_ai.csv')

if __name__ == "__main__":
    num_rows=1000000
    df=run_pureview_ai(num_rows)
    output_pureview(df)