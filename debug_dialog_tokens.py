"""
debug_dialog_tokens.py
======================
Inspect quote-like and dash-like tokens in the annotated corpus.

Purpose
-------
This diagnostic script verifies whether the annotated token table preserves
the punctuation needed for dialog / narration segmentation.

It checks:
  1. Annotated schema
  2. Presence of token_id
  3. Global quote/dash token counts
  4. Quote/dash token counts by document
  5. Ordered token samples using chunk_id + token_id

Input
-----
DATA/processed/corpus_annotated

This script does not write output files.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.spark_session import create_spark_session


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ANNOTATED_DIR = "DATA/processed/corpus_annotated"

# Includes:
#   French angle quotes: « »
#   curly English quotes: “ ”
#   straight double quote: "
#   apostrophe / straight single quote: '
#   em dash / en dash / hyphen: — – -
QUOTE_DASH_PATTERN = r"""[«»“”"'—–-]"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _verify_schema(df: DataFrame) -> None:
    """
    Verify that the annotated corpus has the columns needed for dialog repair.

    token_id is required by the repaired stylometry pipeline because dialog
    segmentation must process tokens in textual order.  If token_id is missing,
    DATA/processed/corpus_annotated was produced by the older annotate.py and
    must be regenerated.
    """
    required_cols = {
        "author",
        "document_id",
        "language",
        "chunk_id",
        "token_id",
        "token",
        "lemma",
        "pos",
        "ner",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            "[debug_dialog_tokens] Annotated corpus is missing required columns: "
            f"{sorted(missing)}. "
            "Re-run annotation with the updated src/annotate.py."
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def show_global_quote_counts(df: DataFrame) -> None:
    """
    Print global counts for all quote-like and dash-like tokens.
    """
    print("[debug_dialog_tokens] Global quote / dash token counts:")

    (
        df
        .where(F.col("token").rlike(QUOTE_DASH_PATTERN))
        .groupBy("token")
        .count()
        .orderBy(F.desc("count"))
        .show(100, truncate=False)
    )


def show_document_quote_counts(df: DataFrame) -> None:
    """
    Print quote/dash counts by author and document.
    """
    print("[debug_dialog_tokens] Quote / dash token counts by document:")

    (
        df
        .where(F.col("token").rlike(QUOTE_DASH_PATTERN))
        .groupBy("author", "document_id", "token")
        .count()
        .orderBy("author", "document_id", F.desc("count"))
        .show(300, truncate=False)
    )


def show_ordered_quote_samples(df: DataFrame) -> None:
    """
    Print ordered quote/dash token samples.

    This confirms that chunk_id + token_id are available and usable as a
    deterministic text-order axis for the repaired dialog segmentation.
    """
    print("[debug_dialog_tokens] Ordered quote / dash token samples:")

    (
        df
        .where(F.col("token").rlike(QUOTE_DASH_PATTERN))
        .select(
            "author",
            "document_id",
            "language",
            "chunk_id",
            "token_id",
            "token",
            "pos",
        )
        .orderBy("author", "document_id", "chunk_id", "token_id")
        .show(300, truncate=False)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    spark = create_spark_session("debug-dialog-tokens")

    try:
        print(f"[debug_dialog_tokens] Loading annotated corpus from {ANNOTATED_DIR}...")
        df = spark.read.parquet(ANNOTATED_DIR)

        print("[debug_dialog_tokens] Annotated schema:")
        df.printSchema()

        _verify_schema(df)

        show_global_quote_counts(df)
        show_document_quote_counts(df)
        show_ordered_quote_samples(df)

        print("[debug_dialog_tokens] Done.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()