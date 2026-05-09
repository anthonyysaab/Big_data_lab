from src.spark_session import create_spark_session
from src.readability import compute_and_save_readability


def main() -> None:
    spark = create_spark_session()
    try:
        compute_and_save_readability(
            spark,
            annotated_dir="DATA/processed/corpus_annotated",
            global_out="DATA/processed/readability_global",
            sliding_out="DATA/processed/readability_sliding",
            window_sizes=[5, 10, 20],
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()