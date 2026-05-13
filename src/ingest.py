"""
src/ingest.py
=============
Load the raw literary corpus into Spark.

Input layout
------------
DATA/raw/{author}/{title}.txt

Output schema
-------------
raw_text, source_path, author, filename, document_id, language, era

Language is assigned mostly by author, with document-level overrides when an
author has mixed-language files.
"""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StringType, StructType


# ---------------------------------------------------------------------------
# Metadata maps
# ---------------------------------------------------------------------------

LANGUAGE_MAP = {
    "balzac": "en",
    "dumas": "en",
    "stendhal": "fr",
    "hugo": "en",
    "sand": "en",       # Current Indiana file is the English translation.
    "flaubert": "en",
    "zola": "en",       # Default for Zola; Nana is overridden below.
    "dickens": "en",
}

DOCUMENT_LANGUAGE_OVERRIDES = {
    ("zola", "nana"): "fr",
}

ERA_MAP = {
    "balzac": "romantic",
    "dumas": "romantic",
    "stendhal": "romantic",
    "hugo": "romantic",
    "sand": "romantic",
    "flaubert": "realist",
    "zola": "realist",
    "dickens": "realist",
}


SCHEMA = StructType(
    [
        StructField("raw_text", StringType(), False),
        StructField("source_path", StringType(), False),
        StructField("author", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("language", StringType(), False),
        StructField("era", StringType(), False),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_language(author: str, document_id: str) -> str:
    """
    Return the correct language for a document.

    Most documents use the author-level language map. Mixed-language cases
    are handled with DOCUMENT_LANGUAGE_OVERRIDES.
    """
    key = (author, document_id)

    if key in DOCUMENT_LANGUAGE_OVERRIDES:
        return DOCUMENT_LANGUAGE_OVERRIDES[key]

    if author not in LANGUAGE_MAP:
        raise ValueError(f"Missing language mapping for author: {author}")

    return LANGUAGE_MAP[author]


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------

def load_text_corpus(spark: SparkSession, data_dir: str = "DATA/raw") -> DataFrame:
    """
    Load all .txt files from DATA/raw into a Spark DataFrame.

    Expected layout:
        DATA/raw/{author}/{title}.txt

    pathlib is used intentionally because it is reliable on Windows.
    """
    base_dir = Path(data_dir)
    txt_files = sorted(base_dir.glob("*/*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")

    rows = []

    for path in txt_files:
        author = path.parent.name.lower()
        filename = path.name
        document_id = path.stem
        source_path = str(path.resolve())
        raw_text = path.read_text(encoding="utf-8")

        if author not in ERA_MAP:
            raise ValueError(f"Missing era mapping for author: {author}")

        rows.append(
            {
                "raw_text": raw_text,
                "source_path": source_path,
                "author": author,
                "filename": filename,
                "document_id": document_id,
                "language": get_language(author, document_id),
                "era": ERA_MAP[author],
            }
        )

    return spark.createDataFrame(rows, schema=SCHEMA)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_corpus(df: DataFrame, output_dir: str, fmt: str = "parquet") -> None:
    """
    Save corpus as Parquet or ORC, partitioned by author.

    This writes exactly to output_dir.
    It does not append _parquet or _orc automatically.
    """
    fmt = fmt.lower()

    if fmt not in {"parquet", "orc"}:
        raise ValueError("fmt must be 'parquet' or 'orc'")

    writer = (
        df.write
        .mode("overwrite")
        .partitionBy("author")
    )

    if fmt == "parquet":
        writer.parquet(output_dir)
    else:
        writer.orc(output_dir)