"""
run_stylometry.py
=================
Standalone entry point to recompute stylometry only.

Requires DATA/processed/corpus_annotated to already exist.

Usage:
    .\.venv\Scripts\python.exe run_stylometry.py
"""

from src.spark_session import create_spark_session
from src.stylometry import compute_and_save_stylometry


def main() -> None:
    spark = create_spark_session()
    try:
        compute_and_save_stylometry(
            spark,
            annotated_dir="DATA/processed/corpus_annotated",
            zipf_out="DATA/processed/stylometry_zipf",
            dialog_out="DATA/processed/stylometry_dialog",
            ttr_out="DATA/processed/stylometry_ttr",
            pos_out="DATA/processed/stylometry_pos",
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
