from src.spark_session import create_spark_session
from src.ingest import load_text_corpus

spark = create_spark_session()

corpus = load_text_corpus(spark, "DATA/raw")

corpus.select(
    "author",
    "document_id",
    "filename",
    "language",
).show(truncate=False)

spark.stop()