from src.spark_session import create_spark_session
from src.ingest import load_text_corpus, save_corpus


spark = create_spark_session()

corpus = load_text_corpus(spark, "DATA/raw")

print(f"Loaded {corpus.count()} documents")
corpus.select("author", "era", "language", "document_id").show(truncate=False)

save_corpus(corpus, "DATA/processed/corpus_parquet", fmt="parquet")
save_corpus(corpus, "DATA/processed/corpus_orc", fmt="orc")

print("Saved parquet → DATA/processed/corpus_parquet")
print("Saved orc     → DATA/processed/corpus_orc")

spark.stop()