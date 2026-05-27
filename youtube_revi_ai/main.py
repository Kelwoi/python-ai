import os
import re
import json
from typing import List, Dict, Any

import pandas as pd
import nltk
import spacy
from dotenv import load_dotenv
from googleapiclient.discovery import build
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")


load_dotenv()


API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YouTube API key was not found. Please add it to the .env file.")


youtube = build("youtube", "v3", developerKey=API_KEY)

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def extract_video_id(url: str) -> str:
    """
    Extracts YouTube video ID from different YouTube URL formats.
    """

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Could not extract video ID from URL.")


def clean_text(text: str) -> str:
    """
    Cleans text by removing links, special symbols and extra spaces.
    """

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_youtube_comments(video_id: str, max_comments: int = 100) -> List[str]:
    """
    Exports comments from a YouTube video using YouTube Data API.
    """

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return comments


def tokenize_text(text: str) -> List[str]:
    """
    Splits text into tokens.
    """

    return nltk.word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Removes common stop words from tokens.
    """

    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Applies stemming to tokens.
    """

    return [stemmer.stem(token) for token in tokens]


def lemmatize_text(text: str) -> List[str]:
    """
    Applies lemmatization using spaCy.
    """

    doc = nlp(text)

    return [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]


def process_comment(comment: str) -> Dict[str, Any]:
    """
    Processes one comment using tokenization, stop-word removal,
    stemming and lemmatization.
    """

    cleaned_text = clean_text(comment)

    tokens = tokenize_text(cleaned_text)
    tokens_without_stopwords = remove_stopwords(tokens)
    stemmed_tokens = stem_tokens(tokens_without_stopwords)
    lemmatized_tokens = lemmatize_text(cleaned_text)

    return {
        "original_comment": comment,
        "cleaned_comment": cleaned_text,
        "tokens": tokens,
        "tokens_without_stopwords": tokens_without_stopwords,
        "stemmed_tokens": stemmed_tokens,
        "lemmatized_tokens": lemmatized_tokens
    }


def save_to_json(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Saves processed data to JSON file.
    """

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def save_to_csv(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Saves processed data to CSV file.
    Lists are converted to strings for better CSV readability.
    """

    csv_ready_data = []

    for item in data:
        csv_ready_data.append({
            "original_comment": item["original_comment"],
            "cleaned_comment": item["cleaned_comment"],
            "tokens": ", ".join(item["tokens"]),
            "tokens_without_stopwords": ", ".join(item["tokens_without_stopwords"]),
            "stemmed_tokens": ", ".join(item["stemmed_tokens"]),
            "lemmatized_tokens": ", ".join(item["lemmatized_tokens"])
        })

    df = pd.DataFrame(csv_ready_data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    Main program function.
    """

    video_url = input("Enter YouTube video URL: ")
    max_comments_input = input("Enter number of comments to export: ")

    try:
        max_comments = int(max_comments_input)
    except ValueError:
        print("Invalid number. Default value 100 will be used.")
        max_comments = 100

    try:
        video_id = extract_video_id(video_url)

        print("Exporting comments from YouTube...")
        comments = get_youtube_comments(video_id, max_comments)

        if not comments:
            print("No comments were found.")
            return

        print("Processing comments with NLP methods...")
        processed_comments = [process_comment(comment) for comment in comments]

        save_to_json(processed_comments, "comments_processed.json")
        save_to_csv(processed_comments, "comments_processed.csv")

        print("Done!")
        print("Files created:")
        print("- comments_processed.json")
        print("- comments_processed.csv")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()