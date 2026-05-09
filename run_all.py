"""
run_all.py
==========
Full project pipeline for the literary big-data analysis.

Pipeline steps:
  1. Load raw text corpus from DATA/raw
  2. Save raw corpus as Parquet and ORC
  3. Clean Project Gutenberg texts
  4. Annotate corpus with Spark NLP
  5. Verify annotated token table
  6. Compute readability indices
  7. Compute stylometric metrics

Input
-----
DATA/raw/{author}/{title}.txt

Outputs
-------
DATA/processed/corpus_parquet
DATA/processed/corpus_orc
DATA/processed/corpus_annotated
DATA/processed/readability_global
DATA/processed/readability_sliding
DATA/processed/stylometry_zipf
DATA/processed/stylometry_dialog
DATA/processed/stylometry_ttr
DATA/processed/stylometry_pos
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.spark_session import create_spark_session
from src.ingest import load_text_corpus, save_corpus
from src.clean import strip_gutenberg_headers, clean_corpus
from src.annotate import annotate_corpus
from src.readability import compute_and_save_readability
from src.stylometry import compute_and_save_stylometry


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = "DATA/raw"

CORPUS_PARQUET_DIR = "DATA/processed/corpus_parquet"
CORPUS_ORC_DIR = "DATA/processed/corpus_orc"
ANNOTATED_DIR = "DATA/processed/corpus_annotated"

READABILITY_GLOBAL_DIR = "DATA/processed/readability_global"
READABILITY_SLIDING_DIR = "DATA/processed/readability_sliding"

STYLOMETRY_ZIPF_DIR = "DATA/processed/stylometry_zipf"
STYLOMETRY_DIALOG_DIR = "DATA/processed/stylometry_dialog"
STYLOMETRY_TTR_DIR = "DATA/processed/stylometry_ttr"
STYLOMETRY_POS_DIR = "DATA/processed/stylometry_pos"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _verify_annotated_schema(df: DataFrame) -> None:
    """
    Verify that the annotated corpus has the columns required downstream.

    token_id is especially important: stylometry dialog segmentation depends
    on a stable token order inside each chunk.  If token_id is missing, the
    annotated corpus was produced with an older version of src/annotate.py and
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
            "[run_all] Annotated corpus is missing required columns: "
            f"{sorted(missing)}. "
            "Re-run annotation with the updated src/annotate.py."
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    spark = create_spark_session("big_data_full_pipeline")

    try:
        print("[run_all] Loading raw corpus...")
        raw_df = load_text_corpus(spark, RAW_DIR)
        print(f"[run_all] Raw document count: {raw_df.count()}")

        print("[run_all] Saving processed Parquet...")
        save_corpus(raw_df, CORPUS_PARQUET_DIR, fmt="parquet")

        print("[run_all] Saving processed ORC...")
        save_corpus(raw_df, CORPUS_ORC_DIR, fmt="orc")

        print("[run_all] Reading processed Parquet...")
        df = spark.read.parquet(CORPUS_PARQUET_DIR)

        print("[run_all] Cleaning corpus...")
        df_clean = clean_corpus(strip_gutenberg_headers(df))

        print("[run_all] Clean corpus statistics:")
        (
            df_clean
            .select(
                "author",
                "document_id",
                "language",
                "era",
                "char_count",
                "word_count",
                "sentence_count",
                "avg_words_per_sentence",
            )
            .orderBy(F.desc("word_count"))
            .show(100, truncate=False)
        )

        print("[run_all] Annotating corpus...")
        annotate_corpus(df_clean, output_dir=ANNOTATED_DIR)

        print("[run_all] Verifying annotated output...")
        annotated = spark.read.parquet(ANNOTATED_DIR)

        _verify_annotated_schema(annotated)

        print("[run_all] Annotated schema:")
        annotated.printSchema()

        print("[run_all] Annotated sample:")
        annotated.show(20, truncate=False)

        print("[run_all] Token rows by author:")
        (
            annotated
            .groupBy("author")
            .count()
            .orderBy(F.desc("count"))
            .show(100, truncate=False)
        )

        print(f"[run_all] Annotated token rows: {annotated.count()}")

        print("[run_all] Computing readability indices...")
        compute_and_save_readability(
            spark,
            annotated_dir=ANNOTATED_DIR,
            global_out=READABILITY_GLOBAL_DIR,
            sliding_out=READABILITY_SLIDING_DIR,
            window_sizes=[5, 10, 20],
        )

        print("[run_all] Computing stylometry...")
        compute_and_save_stylometry(
            spark,
            annotated_dir=ANNOTATED_DIR,
            zipf_out=STYLOMETRY_ZIPF_DIR,
            dialog_out=STYLOMETRY_DIALOG_DIR,
            ttr_out=STYLOMETRY_TTR_DIR,
            pos_out=STYLOMETRY_POS_DIR,
        )

        print("[run_all] Complete.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()