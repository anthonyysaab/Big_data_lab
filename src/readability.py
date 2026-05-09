"""
src/readability.py
==================
Readability indices for the literary corpus.

Computes:
  - Global per-document Flesch-Kincaid (EN) and Kandel-Moles (FR) indices
  - Sliding-window versions over configurable window sizes

Input
-----
Annotated token table:  DATA/processed/corpus_annotated
    columns: document_id, language, chunk_id (int), token, lemma, pos, ner, author

Output
------
DATA/processed/readability_global   (parquet, partitioned by author)
DATA/processed/readability_sliding  (parquet, partitioned by author)

Formulas
--------
Flesch Reading Ease (EN):
    206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)

Kandel-Moles (FR, adapted Flesch for French):
    207.0   - 1.015 * (words / sentences) - 73.6 * (syllables / words)

Syllables: vowel-group heuristic, min 1 per word, language-aware vowel sets.
Sentences: terminal punctuation tokens (., !, ?) detected directly from the
           token string — robust to both UD (FR) and Penn Treebank (EN) POS
           tagsets used by Spark NLP.
"""

import re
from typing import List

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ---------------------------------------------------------------------------
# Syllable counting
# ---------------------------------------------------------------------------

# Vowel groups per language.  French adds accented vowels.
_VOWELS_EN = re.compile(r"[aeiouy]+", re.IGNORECASE)
_VOWELS_FR = re.compile(r"[aeiouyàâäéèêëîïôùûüœæ]+", re.IGNORECASE)

# Tokens that are purely punctuation / numbers should not count as words.
_NOT_A_WORD = re.compile(r"^[\W\d]+$")


def _count_syllables(token: str, lang: str) -> int:
    """
    Heuristic syllable count: number of vowel groups in the token.
    Minimum 1 for any real word token.
    Returns 0 for pure-punctuation / digit-only tokens.
    """
    if not token or _NOT_A_WORD.match(token):
        return 0
    pattern = _VOWELS_FR if lang == "fr" else _VOWELS_EN
    groups = pattern.findall(token)
    return max(1, len(groups))


def _syllable_udf_factory():
    """
    Return a UDF that takes (token: str, language: str) -> int.
    Defined inside a factory so the closure captures only pure Python objects.
    """
    import re

    vowels_en = re.compile(r"[aeiouy]+", re.IGNORECASE)
    vowels_fr = re.compile(r"[aeiouyàâäéèêëîïôùûüœæ]+", re.IGNORECASE)
    not_a_word = re.compile(r"^[\W\d]+$")

    def _syllables(token, lang):
        if not token or not_a_word.match(token):
            return 0
        pattern = vowels_fr if lang == "fr" else vowels_en
        groups = pattern.findall(token)
        return max(1, len(groups))

    return F.udf(_syllables, T.IntegerType())


_syllable_udf = _syllable_udf_factory()


# ---------------------------------------------------------------------------
# Sentence boundary detection from token string
# ---------------------------------------------------------------------------

def _is_sentence_end_udf():
    """
    Detect sentence-final tokens by matching the token string directly.

    Why not use POS:
    - French Spark NLP uses Universal Dependencies tags: PUNCT
    - English Spark NLP uses Penn Treebank tags: the tag for '.' is literally
      '.' not 'PUNCT', and '!' / '?' are tagged similarly.
    Matching on the token string itself is correct for both tagsets because
    Spark NLP's Tokenizer isolates terminal punctuation as standalone tokens
    in both FR and EN pipelines.
    """
    sent_end = {".", "!", "?", "…", "...", "?!", "!?"}

    def _is_end(token, pos):
        if token in sent_end:
            return 1
        return 0

    return F.udf(_is_end, T.IntegerType())


_sent_end_udf = _is_sentence_end_udf()


# ---------------------------------------------------------------------------
# Core aggregation: per-chunk stats
# ---------------------------------------------------------------------------

def _per_chunk_stats(token_df: DataFrame) -> DataFrame:
    """
    Aggregate token-level data to per-(author, document_id, chunk_id) stats:
      word_count, syllable_count, sentence_count

    A 'word' is any token whose POS is not PUNCT, SYM, or X and is not a
    pure-punctuation string.
    """
    return (
        token_df
        .withColumn("syllables", _syllable_udf(F.col("token"), F.col("language")))
        .withColumn(
            "is_word",
            F.when(
                F.col("pos").isin("PUNCT", "SYM", "X") | (F.col("syllables") == 0),
                F.lit(0),
            ).otherwise(F.lit(1)),
        )
        .withColumn(
            "is_sent_end",
            _sent_end_udf(F.col("token"), F.col("pos")),
        )
        .groupBy("author", "document_id", "language", "chunk_id")
        .agg(
            F.sum("is_word").cast("long").alias("word_count"),
            F.sum("syllables").cast("long").alias("syllable_count"),
            # +1 so that the last sentence (no trailing punctuation) is counted
            (F.sum("is_sent_end") + F.lit(1)).cast("long").alias("sentence_count"),
        )
    )


# ---------------------------------------------------------------------------
# Readability formula
# ---------------------------------------------------------------------------

def _readability_from_agg(
    agg_df: DataFrame,
    word_col: str = "word_count",
    syl_col: str = "syllable_count",
    sent_col: str = "sentence_count",
    score_col: str = "readability_score",
    index_col: str = "readability_index",
) -> DataFrame:
    """
    Compute readability score from aggregated word/syllable/sentence counts.

    For English  → Flesch Reading Ease
    For French   → Kandel-Moles

    Both return a score roughly in [0, 100]:
      < 30   very difficult
      30-50  difficult
      50-60  fairly difficult
      60-70  standard
      > 70   easy
    """
    words = F.col(word_col).cast("double")
    syls  = F.col(syl_col).cast("double")
    sents = F.col(sent_col).cast("double")

    # Guard: avoid division by zero
    safe_sents = F.greatest(sents, F.lit(1.0))
    safe_words = F.greatest(words, F.lit(1.0))

    avg_sent_len = words / safe_sents          # words per sentence
    avg_syl_word = syls  / safe_words          # syllables per word

    flesch = F.lit(206.835) - F.lit(1.015) * avg_sent_len - F.lit(84.6) * avg_syl_word
    kandel = F.lit(207.0)   - F.lit(1.015) * avg_sent_len - F.lit(73.6) * avg_syl_word

    score  = F.when(F.col("language") == "fr", kandel).otherwise(flesch)
    index  = F.when(F.col("language") == "fr", F.lit("Kandel-Moles")).otherwise(F.lit("Flesch-Kincaid"))

    return (
        agg_df
        .withColumn(score_col, score)
        .withColumn(index_col, index)
    )


# ---------------------------------------------------------------------------
# Global readability (one score per document)
# ---------------------------------------------------------------------------

def compute_global_readability(token_df: DataFrame) -> DataFrame:
    """
    Return one readability row per (author, document_id).

    Output columns:
        author, document_id, language,
        word_count, syllable_count, sentence_count,
        avg_sentence_length, avg_syllables_per_word,
        readability_score, readability_index
    """
    chunk_stats = _per_chunk_stats(token_df)

    doc_stats = (
        chunk_stats
        .groupBy("author", "document_id", "language")
        .agg(
            F.sum("word_count").alias("word_count"),
            F.sum("syllable_count").alias("syllable_count"),
            F.sum("sentence_count").alias("sentence_count"),
        )
    )

    return (
        _readability_from_agg(doc_stats)
        .withColumn(
            "avg_sentence_length",
            (F.col("word_count") / F.greatest(F.col("sentence_count"), F.lit(1))).cast("double"),
        )
        .withColumn(
            "avg_syllables_per_word",
            (F.col("syllable_count") / F.greatest(F.col("word_count"), F.lit(1))).cast("double"),
        )
        .orderBy("author", "document_id")
    )


# ---------------------------------------------------------------------------
# Sliding-window readability
# ---------------------------------------------------------------------------

def compute_sliding_readability(
    token_df: DataFrame,
    window_sizes: List[int] = (5, 10, 20),
) -> DataFrame:
    """
    Compute readability over sliding windows of chunks.

    For each document, chunk_id acts as the position axis (each chunk ~4000
    chars).  A window of size W centred on chunk c aggregates chunks
    [c - W//2, c + W//2].

    Uses Spark's rangeBetween window function — pure Spark, no collect().

    Output columns:
        author, document_id, language, chunk_id,
        window_size,
        word_count, syllable_count, sentence_count,
        readability_score, readability_index
    """
    chunk_stats = _per_chunk_stats(token_df).cache()

    frames = []

    for w in window_sizes:
        half = w // 2

        doc_window = (
            Window
            .partitionBy("author", "document_id")
            .orderBy("chunk_id")
            .rangeBetween(-half, half)
        )

        windowed = (
            chunk_stats
            .withColumn("word_count",     F.sum("word_count").over(doc_window))
            .withColumn("syllable_count", F.sum("syllable_count").over(doc_window))
            .withColumn("sentence_count", F.sum("sentence_count").over(doc_window))
            .withColumn("window_size", F.lit(w))
        )

        scored = _readability_from_agg(windowed)
        frames.append(scored)

    result = frames[0]
    for frame in frames[1:]:
        result = result.union(frame)

    chunk_stats.unpersist()

    return result.orderBy("author", "document_id", "window_size", "chunk_id")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_readability(df: DataFrame, output_dir: str) -> None:
    """
    Save a readability DataFrame partitioned by author.
    Overwrites any existing output.
    """
    (
        df
        .write
        .mode("overwrite")
        .partitionBy("author")
        .parquet(output_dir)
    )


# ---------------------------------------------------------------------------
# Convenience entry point (called from run_all.py or standalone)
# ---------------------------------------------------------------------------

def compute_and_save_readability(
    spark: SparkSession,
    annotated_dir: str = "DATA/processed/corpus_annotated",
    global_out:    str = "DATA/processed/readability_global",
    sliding_out:   str = "DATA/processed/readability_sliding",
    window_sizes:  List[int] = (5, 10, 20),
) -> None:
    """
    Full readability pipeline:
      1. Load annotated corpus
      2. Compute global per-document scores
      3. Compute sliding-window scores
      4. Save both to Parquet
    """
    print("[readability] Loading annotated corpus...")
    token_df = spark.read.parquet(annotated_dir).cache()
    print(f"[readability] Token rows: {token_df.count()}")

    print("[readability] Computing global readability scores...")
    global_df = compute_global_readability(token_df)
    global_df.show(20, truncate=False)
    save_readability(global_df, global_out)
    print(f"[readability] Global scores saved to {global_out}")

    print(f"[readability] Computing sliding readability (windows={list(window_sizes)})...")
    sliding_df = compute_sliding_readability(token_df, window_sizes=window_sizes)
    save_readability(sliding_df, sliding_out)
    print(f"[readability] Sliding scores saved to {sliding_out}")

    token_df.unpersist()
    print("[readability] Done.")