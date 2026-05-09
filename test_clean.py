from pyspark.sql import functions as F

from src.spark_session import create_spark_session
from src.ingest import load_text_corpus
from src.clean import strip_gutenberg_headers, clean_corpus


spark = create_spark_session()

corpus = load_text_corpus(spark, "DATA/raw")

cleaned = clean_corpus(strip_gutenberg_headers(corpus))

(
    cleaned
    .select(
        "author",
        "document_id",
        "char_count",
        "word_count",
        "sentence_count",
        "avg_words_per_sentence",
    )
    .orderBy(F.desc("word_count"))
    .show(100, truncate=False)
)

spark.stop()