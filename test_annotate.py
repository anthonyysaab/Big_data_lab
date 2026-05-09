from src.spark_session import create_spark_session
from src.clean import clean_corpus, strip_gutenberg_headers
from src.annotate import annotate_corpus


spark = create_spark_session()

df = spark.read.parquet("DATA/processed/corpus_parquet")
df = df.filter(df.document_id == "indiana")
df = strip_gutenberg_headers(df)
df_clean = clean_corpus(df)

annotate_corpus(df_clean, output_dir="DATA/processed/corpus_annotated_test")

spark.stop()