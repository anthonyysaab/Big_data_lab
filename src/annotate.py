from pathlib import Path
import shutil
from typing import List

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from sparknlp.annotator import (
    SentenceDetector,
    Tokenizer,
    LemmatizerModel,
    PerceptronModel,
)
from sparknlp.base import DocumentAssembler, Finisher


CHUNK_MAX_CHARS = 4000


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str | None, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """
    Split long novels into smaller chunks before Spark NLP annotation.

    This keeps each Spark NLP row small enough for a local Windows machine.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        extra_len = len(word) + 1

        if current_words and current_len + extra_len > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)
        else:
            current_words.append(word)
            current_len += extra_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


_chunk_text_udf = F.udf(_chunk_text, T.ArrayType(T.StringType()))


def _prepare_chunks(df: DataFrame) -> DataFrame:
    required_cols = {"document_id", "author", "language", "text_clean"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"annotate_corpus missing required columns: {sorted(missing)}")

    return (
        df
        .withColumn("chunk_texts", _chunk_text_udf(F.col("text_clean")))
        .select(
            "document_id",
            "author",
            "language",
            F.posexplode(F.col("chunk_texts")).alias("chunk_id", "text_clean"),
        )
    )


# ---------------------------------------------------------------------------
# French pipeline: token + lemma + POS
# ---------------------------------------------------------------------------

def _build_fr_pipeline() -> Pipeline:
    document = (
        DocumentAssembler()
        .setInputCol("text_clean")
        .setOutputCol("document")
    )

    sentence = (
        SentenceDetector()
        .setInputCols(["document"])
        .setOutputCol("sentence")
    )

    tokenizer = (
        Tokenizer()
        .setInputCols(["sentence"])
        .setOutputCol("token")
    )

    lemma = (
        LemmatizerModel.pretrained("lemma_spacylookup", "fr")
        .setInputCols(["token"])
        .setOutputCol("lemma")
    )

    pos = (
        PerceptronModel.pretrained("pos_ud_gsd", "fr")
        .setInputCols(["sentence", "token"])
        .setOutputCol("pos")
    )

    finisher = (
        Finisher()
        .setInputCols(["token", "lemma", "pos"])
        .setOutputCols(["token_out", "lemma_out", "pos_out"])
        .setOutputAsArray(True)
        .setCleanAnnotations(False)
    )

    return Pipeline(stages=[document, sentence, tokenizer, lemma, pos, finisher])


# ---------------------------------------------------------------------------
# English pipeline: token + lemma + POS
# ---------------------------------------------------------------------------

def _build_en_pipeline() -> Pipeline:
    document = (
        DocumentAssembler()
        .setInputCol("text_clean")
        .setOutputCol("document")
    )

    sentence = (
        SentenceDetector()
        .setInputCols(["document"])
        .setOutputCol("sentence")
    )

    tokenizer = (
        Tokenizer()
        .setInputCols(["sentence"])
        .setOutputCol("token")
    )

    lemma = (
        LemmatizerModel.pretrained("lemma_antbnc", "en")
        .setInputCols(["token"])
        .setOutputCol("lemma")
    )

    pos = (
        PerceptronModel.pretrained("pos_anc", "en")
        .setInputCols(["sentence", "token"])
        .setOutputCol("pos")
    )

    finisher = (
        Finisher()
        .setInputCols(["token", "lemma", "pos"])
        .setOutputCols(["token_out", "lemma_out", "pos_out"])
        .setOutputAsArray(True)
        .setCleanAnnotations(False)
    )

    return Pipeline(stages=[document, sentence, tokenizer, lemma, pos, finisher])


# ---------------------------------------------------------------------------
# Explode token-level rows
# ---------------------------------------------------------------------------

def _explode_annotations(df: DataFrame) -> DataFrame:
    """
    Convert Spark NLP finished arrays into one row per token.

    Adds:
    - token_id: token position inside each chunk

    Why token_id matters:
    Downstream stylometric analyses such as dialog / narration segmentation
    need a stable textual order. Spark row order is not reliable after
    transformations or Parquet writes, so we preserve the token position at
    annotation time using posexplode().

    ner is a lightweight entity-candidate label:
    - ENT for proper nouns
    - O otherwise

    This avoids the huge deep-learning NER embedding model that caused the
    Java heap crash on the full French corpus.
    """
    zipped = df.withColumn(
        "zipped",
        F.arrays_zip("token_out", "lemma_out", "pos_out"),
    )

    exploded = (
        zipped
        .select(
            "document_id",
            "author",
            "language",
            "chunk_id",
            F.posexplode(F.col("zipped")).alias("token_id", "zipped"),
        )
        .select(
            "document_id",
            "author",
            "language",
            "chunk_id",
            F.col("token_id").cast("int").alias("token_id"),
            F.col("zipped.token_out").alias("token"),
            F.col("zipped.lemma_out").alias("lemma"),
            F.col("zipped.pos_out").alias("pos"),
        )
    )

    return (
        exploded
        .withColumn(
            "ner",
            F.when(
                F.col("pos").isin("PROPN", "NNP", "NNPS"),
                F.lit("ENT"),
            ).otherwise(F.lit("O")),
        )
    )


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_document(token_df: DataFrame, out_path: Path, author: str) -> None:
    """
    Write one document at a time into Hive-style author partitions.

    Spark can read the root directory later and recover author from:
        author=<name>
    """
    target = out_path / f"author={author}"
    target.mkdir(parents=True, exist_ok=True)

    (
        token_df
        .drop("author")
        .coalesce(1)
        .write
        .mode("append")
        .parquet(str(target))
    )


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------

def annotate_corpus(
    df: DataFrame,
    output_dir: str = "DATA/processed/corpus_annotated",
) -> None:
    """
    Annotate the corpus with token, lemma, POS, and lightweight entity labels.

    Output layout:
        DATA/processed/corpus_annotated/author=<author>/*.parquet

    This version is designed to complete on a local Windows laptop.
    """
    out_path = Path(output_dir)

    if out_path.exists():
        shutil.rmtree(out_path)

    out_path.mkdir(parents=True, exist_ok=True)

    chunked = _prepare_chunks(df)

    wrote_anything = False

    for lang, build_pipeline in [
        ("fr", _build_fr_pipeline),
        ("en", _build_en_pipeline),
    ]:
        lang_chunks = chunked.filter(F.col("language") == lang)

        if lang_chunks.rdd.isEmpty():
            continue

        doc_plan = (
            lang_chunks
            .groupBy("language", "author", "document_id")
            .agg(F.count("*").alias("chunk_count"))
            .orderBy("author", "document_id")
            .collect()
        )

        print(
            f"[annotate] Building {lang.upper()} pipeline for "
            f"{len(doc_plan)} documents..."
        )

        pipeline = build_pipeline()
        model = pipeline.fit(lang_chunks.limit(1))

        for row in doc_plan:
            author = row["author"]
            document_id = row["document_id"]
            chunk_count = row["chunk_count"]

            print(
                f"[annotate] {lang.upper()} document: "
                f"{author}/{document_id} ({chunk_count} chunks)"
            )

            doc_chunks = lang_chunks.filter(
                (F.col("author") == author)
                & (F.col("document_id") == document_id)
            )

            annotated = model.transform(doc_chunks)
            token_df = _explode_annotations(annotated)

            _write_document(token_df, out_path, author)
            wrote_anything = True

    if not wrote_anything:
        raise RuntimeError("No documents were annotated.")

    print(f"[annotate] Done. Output saved to {out_path}")