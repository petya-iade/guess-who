"""
Reddit topic modeling with BERTopic.

Loads JSONL files from data/raw, cleans the text, trains BERTopic,
and writes the results to data/processed.

"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

# Resolve paths relative to this script, not the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

MAX_DOCUMENTS = 5_000       # Set to None to use all usable documents.
MIN_WORDS = 8
RANDOM_STATE = 42
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# Terms that are usually unhelpful in Reddit topic labels.
CUSTOM_STOPWORDS = {
    "reddit", "subreddit", "subreddits", "karma",
    "upvote", "upvotes", "downvote", "downvotes",
    "www", "com", "http", "https", "amp",
    "don", "doesn", "didn", "isn", "aren", "wasn",
    "ve", "ll", "im", "ive", "dont", "doesnt", "didnt",
    "cant", "couldnt", "wouldnt", "youre", "theyre",
    "thats", "theres",
    "just", "like", "really", "also", "get", "got",
    "going", "would", "could", "will", "one", "anyone",
    "people", "know", "think", "want", "say", "said",
    "make", "way", "time", "thing", "things", "something",
    "every", "see", "much", "many", "even", "still",
    "thanks", "thank", "you", "me", "hi", "hello",
    "use", "used", "find", "able", "click", "posts",
    "post", "comment", "comments", "advice", "message",
    "question", "questions", "read", "lot", "better",
    "worse", "worst", "need", "removed", "did",
}


# ---------------------------------------------------------------------
# 2. PREPARE OUTPUT DIRECTORY
# ---------------------------------------------------------------------

def reset_output_folder(folder: Path) -> None:
    """Delete the previous processed output and create an empty folder."""
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder reset: {folder}")


# ---------------------------------------------------------------------
# 3. LOAD JSONL FILES
# ---------------------------------------------------------------------

def load_jsonl_files(folder: Path) -> pd.DataFrame:
    """Load dictionary records from every JSONL file in a folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {folder}")

    jsonl_files = sorted(folder.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in: {folder}")

    print(f"Found {len(jsonl_files)} JSONL file(s).")
    records = []

    for file_path in jsonl_files:
        print(f"Loading: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    print(
                        f"Skipping invalid JSON in {file_path.name}, "
                        f"line {line_number}: {error}"
                    )
                    continue

                if isinstance(record, dict):
                    records.append(record)
                else:
                    print(
                        f"Skipping non-object record in {file_path.name}, "
                        f"line {line_number}"
                    )

    if not records:
        raise ValueError("No valid JSON object records were found.")

    dataframe = pd.DataFrame(records)
    print(f"\nTotal records loaded: {len(dataframe):,}")
    print(f"Available columns: {dataframe.columns.tolist()}")

    return dataframe


# ---------------------------------------------------------------------
# 4. EXTRACT AND CLEAN TEXT
# ---------------------------------------------------------------------

def select_text_column(dataframe: pd.DataFrame) -> str:
    """Choose the available text field."""
    for column in ("body", "selftext", "text"):
        if column in dataframe.columns:
            return column

    raise ValueError("No 'body', 'selftext', or 'text' column was found.")


def clean_text(value: object) -> str:
    """Remove common Reddit markup and normalize whitespace."""
    text = html.unescape(str(value))

    # Remove URLs, Reddit user mentions, and subreddit mentions.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu/\w+|\br/\w+", " ", text, flags=re.IGNORECASE)

    # Remove Markdown quote lines.
    text = re.sub(r"(?m)^\s*>.*$", " ", text)

    # Normalize whitespace.
    return re.sub(r"\s+", " ", text).strip()


def prepare_documents(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Combine titles and text, clean them, and remove unusable records."""
    body_column = select_text_column(dataframe)

    bodies = dataframe[body_column].fillna("").astype(str)

    if "title" in dataframe.columns:
        titles = dataframe["title"].fillna("").astype(str)
    else:
        titles = pd.Series("", index=dataframe.index, dtype=str)

    dataframe = dataframe.copy()
    dataframe["raw_model_text"] = (titles + " " + bodies).str.strip()
    dataframe["model_text"] = dataframe["raw_model_text"].apply(clean_text)

    word_counts = dataframe["model_text"].str.split().str.len()
    placeholders = {"", "nan", "[deleted]", "[removed]"}

    valid_mask = (
        ~dataframe["model_text"].str.lower().isin(placeholders)
        & (word_counts >= MIN_WORDS)
    )

    model_dataframe = dataframe.loc[valid_mask].copy()

    print(f"\nRecords with usable text: {len(model_dataframe):,}")

    if len(model_dataframe) < 20:
        raise ValueError(
            "Fewer than 20 usable documents were found. "
            "Check the text fields or lower MIN_WORDS."
        )

    return model_dataframe


# ---------------------------------------------------------------------
# 5. SAMPLE DOCUMENTS
# ---------------------------------------------------------------------

def sample_documents(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply reproducible sampling if MAX_DOCUMENTS is set."""
    if MAX_DOCUMENTS is not None and len(dataframe) > MAX_DOCUMENTS:
        dataframe = dataframe.sample(
            n=MAX_DOCUMENTS,
            random_state=RANDOM_STATE,
        )

    dataframe = dataframe.reset_index(drop=True)

    print(f"Documents being modeled: {len(dataframe):,}")
    print("\nExample cleaned text:")
    print(dataframe.loc[0, "model_text"][:500])

    return dataframe


# ---------------------------------------------------------------------
# 6. BUILD AND TRAIN BERTopic
# ---------------------------------------------------------------------

def build_topic_model() -> BERTopic:
    """Configure the embedding, dimensionality reduction, and clustering."""
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    english_stopwords = CountVectorizer(
        stop_words="english"
    ).get_stop_words()

    all_stopwords = sorted(set(CUSTOM_STOPWORDS) | set(english_stopwords))

    vectorizer_model = CountVectorizer(
        stop_words=all_stopwords,
        min_df=2,
        ngram_range=(1, 2),
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )


def train_model(
    topic_model: BERTopic,
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic and attach the resulting topic IDs to each document."""
    documents = dataframe["model_text"].tolist()

    print("\nTraining BERTopic...")
    topics, _ = topic_model.fit_transform(documents)

    dataframe = dataframe.copy()
    dataframe["topic"] = topics

    print("\nTopic modeling complete.")
    return dataframe, topic_model.get_topic_info()


# ---------------------------------------------------------------------
# 7. DISPLAY TOPIC SUMMARY
# ---------------------------------------------------------------------

def display_topics(
    topic_model: BERTopic,
    topic_info: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> None:
    print("\n--- TOP 20 TOPICS ---")
    print(topic_info.head(20).to_string(index=False))

    print("\n--- REPRESENTATIVE KEYWORDS AND EXAMPLES ---")

    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue

        keywords = topic_model.get_topic(topic_id)
        if not keywords:
            continue

        print(f"\nTopic {topic_id}:")
        print(", ".join(word for word, _ in keywords[:10]))

        examples = dataframe.loc[
            dataframe["topic"] == topic_id,
            "model_text",
        ].head(3)

        for example in examples:
            print(f"  - {example[:300]}")


# ---------------------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------------------

def save_results(
    topic_model: BERTopic,
    dataframe: pd.DataFrame,
    topic_info: pd.DataFrame,
) -> None:
    posts_path = OUTPUT_FOLDER / "reddit_comments_with_topics.csv"
    topics_path = OUTPUT_FOLDER / "reddit_topic_summary.csv"
    model_path = OUTPUT_FOLDER / "bertopic_model"

    dataframe.to_csv(posts_path, index=False, encoding="utf-8-sig")
    topic_info.to_csv(topics_path, index=False, encoding="utf-8-sig")

    topic_model.save(
        str(model_path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=f"sentence-transformers/{EMBEDDING_MODEL_NAME}",
    )

    print("\nSaved files:")
    print(f"- {posts_path}")
    print(f"- {topics_path}")
    print(f"- {model_path}")


# ---------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------

def main() -> None:
    # This intentionally removes all previous contents of data/processed.
    reset_output_folder(OUTPUT_FOLDER)

    dataframe = load_jsonl_files(DATA_FOLDER)
    dataframe = prepare_documents(dataframe)
    dataframe = sample_documents(dataframe)

    topic_model = build_topic_model()
    dataframe, topic_info = train_model(topic_model, dataframe)

    display_topics(topic_model, topic_info, dataframe)
    save_results(topic_model, dataframe, topic_info)

    print("\nAll done!")


if __name__ == "__main__":
    main()
