import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    flags=re.IGNORECASE | re.DOTALL,
)

_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_gutenberg(text: str | None) -> str:
    """
    Remove Project Gutenberg header/footer from one document string.
    """
    if text is None:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start_match = _START_RE.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = _END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()


_strip_gutenberg_udf = F.udf(_strip_gutenberg, T.StringType())


def strip_gutenberg_headers(df: DataFrame) -> DataFrame:
    """
    Remove Project Gutenberg boilerplate.

    This overwrites raw_text intentionally, so downstream code can still use
    raw_text without remembering a second column name.
    """
    return df.withColumn("raw_text", _strip_gutenberg_udf(F.col("raw_text")))


def clean_corpus(df: DataFrame) -> DataFrame:
    """
    Add cleaned text and document-level statistics.

    Output columns:
    - text_clean
    - char_count
    - word_count
    - sentences
    - sentence_count
    - avg_words_per_sentence

    Important:
    - We do NOT lowercase text_clean because NER benefits from capitalization.
    """
    cleaned = (
        df
        .withColumn("text_clean", F.regexp_replace(F.col("raw_text"), r"\r\n|\r", "\n"))
        .withColumn("text_clean", F.regexp_replace(F.col("text_clean"), r"\n{2,}", "\n\n"))
        .withColumn("text_clean", F.regexp_replace(F.col("text_clean"), r"[ \t]+", " "))
        .withColumn("text_clean", F.trim(F.col("text_clean")))
        .withColumn("char_count", F.length(F.col("text_clean")))
        .withColumn(
            "word_count",
            F.when(
                F.length(F.trim(F.col("text_clean"))) == 0,
                F.lit(0),
            ).otherwise(
                F.size(F.split(F.col("text_clean"), r"\s+"))
            ),
        )
        .withColumn(
            "sentences",
            F.when(
                F.length(F.trim(F.col("text_clean"))) == 0,
                F.array().cast(T.ArrayType(T.StringType())),
            ).otherwise(
                F.split(F.col("text_clean"), r"(?<=[.!?])\s+")
            ),
        )
        .withColumn(
            "sentence_count",
            F.when(
                F.length(F.trim(F.col("text_clean"))) == 0,
                F.lit(0),
            ).otherwise(
                F.size(F.col("sentences"))
            ),
        )
        .withColumn(
            "avg_words_per_sentence",
            F.when(
                F.col("sentence_count") == 0,
                F.lit(0.0),
            ).otherwise(
                (F.col("word_count") / F.col("sentence_count")).cast("double")
            ),
        )
    )

    return cleaned