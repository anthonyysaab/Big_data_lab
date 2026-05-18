"""
src/stylometry.py
=================
Stylometric analysis of the literary corpus.

Computes:
  1. Zipf data         — token frequency rank vs frequency per document
  2. Dialog/narration — per-document dialog ratio
  3. Type-token ratio — unique lemmas / total tokens per document
  4. POS distribution — proportion of NOUN/VERB/ADJ/PROPN per author & document

Input
-----
Annotated token table: DATA/processed/corpus_annotated
    columns:
        document_id, language, chunk_id, token_id,
        token, lemma, pos, ner, author

Output
------
DATA/processed/stylometry_zipf          (parquet, partitioned by author)
DATA/processed/stylometry_dialog        (parquet, partitioned by author)
DATA/processed/stylometry_ttr           (parquet, partitioned by author)
DATA/processed/stylometry_pos           (parquet, partitioned by author)

All outputs are small enough to collect() and plot in a Quarto notebook.
"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_columns(df: DataFrame, required_cols: set[str], context: str) -> None:
    """
    Fail early with a clear error if an input DataFrame is missing columns.

    This is especially important for stylometry because dialog segmentation
    depends on token order. Older annotated corpora do not contain token_id
    and must be regenerated with the updated src/annotate.py.
    """
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{context} missing required columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# 1. Zipf analysis
# ---------------------------------------------------------------------------

def compute_zipf(token_df: DataFrame) -> DataFrame:
    """
    Compute Zipf rank-frequency data per document.

    For each (author, document_id), count how many times each token appears,
    then rank tokens by descending frequency. Zipf's law predicts:
        frequency ∝ 1 / rank

    Output columns:
        author, document_id, language, token,
        frequency, rank, log_rank, log_frequency
    """
    _require_columns(
        token_df,
        {"author", "document_id", "language", "token", "pos"},
        "compute_zipf",
    )

    token_counts = (
        token_df
        .filter(~F.col("pos").isin("PUNCT", "SYM", "X"))
        .withColumn("token_lower", F.lower(F.col("token")))
        .groupBy("author", "document_id", "language", "token_lower")
        .agg(F.count("*").cast("long").alias("frequency"))
    )

    doc_window = (
        Window
        .partitionBy("author", "document_id")
        .orderBy(F.desc("frequency"))
    )

    return (
        token_counts
        .withColumn("rank", F.rank().over(doc_window))
        .withColumn("log_rank", F.log(F.col("rank").cast("double")))
        .withColumn("log_frequency", F.log(F.col("frequency").cast("double")))
        .withColumnRenamed("token_lower", "token")
        .orderBy("author", "document_id", "rank")
    )


# ---------------------------------------------------------------------------
# 2. Dialog vs narration segmentation
# ---------------------------------------------------------------------------

_QUOTE_OPEN_TOKENS = {"«", "“"}
_QUOTE_CLOSE_TOKENS = {"»", "”"}
_QUOTE_TOGGLE_TOKENS = {'"'}

# Dialogue forms observed in the current corpus:
#   "Text"
#   “Text”
#   «Text»
#   -- Text
#   --Text
#   :-- Text
#   — Text / – Text / ― Text
_DASH_DIALOG_TOKENS = {"—", "–", "-", "―", "--", ":--", ":—", ":–", ":―"}

_SENTENCE_END_TOKENS = {".", "!", "?", "…", "?!", "!?"}
_DASH_START_BOUNDARY_TOKENS = _SENTENCE_END_TOKENS | {":", ";", "»", "”", '"'}

_WORDLIKE_RE = r"[A-Za-zÀ-ÖØ-öø-ÿ]"
_DASH_PREFIX_RE = r"^[:;]?(?:--|—|–|-|―)"
_DASH_PREFIX_WORD_RE = r"^[:;]?(?:--|—|–|-|―)[A-Za-zÀ-ÖØ-öø-ÿ]"
_COLON_DASH_RE = r"^[:;](?:--|—|–|-|―)"


def compute_dialog(token_df: DataFrame) -> DataFrame:
    """
    Label each token as 'dialog' or 'narration'.

    Strategy:
    1. Quotation-marker detection:
       - « and “ open dialog
       - » and ” close dialog
       - " toggles dialog state

    2. Dash-dialog detection:
       - handles -- Text
       - handles --Text, e.g. --J'aime
       - handles :-- Text
       - handles em/en/horizontal dash variants: —, –, ―
       - handles cases where -- is tokenised as two separate - tokens

    Important:
    - Mid-word double hyphens such as eyesight--a or think.--For are not
      treated as dialogue openers because they do not start the token.
    - True line-start dialogue should ideally be detected before tokenisation.
      This token-level method is a conservative approximation.

    Output columns:
        author, document_id, language, chunk_id, token_id,
        token, lemma, pos, ner, segment
    """
    _require_columns(
        token_df,
        {
            "author",
            "document_id",
            "language",
            "chunk_id",
            "token_id",
            "token",
            "lemma",
            "pos",
            "ner",
        },
        "compute_dialog",
    )

    open_set = F.array(*[F.lit(q) for q in sorted(_QUOTE_OPEN_TOKENS)])
    close_set = F.array(*[F.lit(q) for q in sorted(_QUOTE_CLOSE_TOKENS)])
    toggle_set = F.array(*[F.lit(q) for q in sorted(_QUOTE_TOGGLE_TOKENS)])
    dash_set = F.array(*[F.lit(q) for q in sorted(_DASH_DIALOG_TOKENS)])
    quote_any = sorted(_QUOTE_OPEN_TOKENS | _QUOTE_CLOSE_TOKENS | _QUOTE_TOGGLE_TOKENS)

    doc_order_window = (
        Window
        .partitionBy("author", "document_id")
        .orderBy("chunk_id", "token_id")
    )

    quote_window = (
        Window
        .partitionBy("author", "document_id", "chunk_id")
        .orderBy("token_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    dash_window = (
        Window
        .partitionBy("author", "document_id")
        .orderBy("chunk_id", "token_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    labeled = (
        token_df
        .withColumn("_doc_pos", F.row_number().over(doc_order_window).cast("long"))
        .withColumn("_prev_token", F.lag("token").over(doc_order_window))
        .withColumn("_next_token", F.lead("token").over(doc_order_window))
        .withColumn("_next2_token", F.lead("token", 2).over(doc_order_window))
        .withColumn("_next3_token", F.lead("token", 3).over(doc_order_window))

        # Quotation markers.
        .withColumn(
            "_is_open_quote",
            F.when(
                F.array_contains(open_set, F.col("token"))
                | F.col("token").rlike(r'[«\u201c]'),
                F.lit(1)
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "_is_close_quote",
            F.when(
                F.array_contains(close_set, F.col("token"))
                | F.col("token").rlike(r'[»\u201d]'),
                F.lit(1)
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "_is_toggle_quote",
            F.when(
                F.array_contains(toggle_set, F.col("token"))
                | F.col("token").rlike(r'^".|."$'),
                F.lit(1)
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "_quote_delta",
            F.col("_is_open_quote") - F.col("_is_close_quote"),
        )

        # Dash/dialog-marker features.
        .withColumn(
            "_is_dash_token",
            F.when(F.array_contains(dash_set, F.col("token")), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "_prev_is_boundary",
            (
                F.col("_prev_token").isNull()
                | F.col("_prev_token").isin(*sorted(_DASH_START_BOUNDARY_TOKENS))
            ),
        )
        .withColumn(
            "_next_is_word",
            F.coalesce(F.col("_next_token").rlike(_WORDLIKE_RE), F.lit(False)),
        )
        .withColumn(
            "_next2_is_word",
            F.coalesce(F.col("_next2_token").rlike(_WORDLIKE_RE), F.lit(False)),
        )
        .withColumn(
            "_next3_is_word",
            F.coalesce(F.col("_next3_token").rlike(_WORDLIKE_RE), F.lit(False)),
        )
        .withColumn(
            "_next_is_quote",
            F.coalesce(F.col("_next_token").isin(*quote_any), F.lit(False)),
        )
        .withColumn(
            "_next2_is_quote",
            F.coalesce(F.col("_next2_token").isin(*quote_any), F.lit(False)),
        )
        .withColumn(
            "_token_starts_dash",
            F.coalesce(F.col("token").rlike(_DASH_PREFIX_RE), F.lit(False)),
        )
        .withColumn(
            "_token_starts_dash_word",
            F.coalesce(F.col("token").rlike(_DASH_PREFIX_WORD_RE), F.lit(False)),
        )
        .withColumn(
            "_token_starts_colon_dash",
            F.coalesce(F.col("token").rlike(_COLON_DASH_RE), F.lit(False)),
        )

        # Examples covered:
        #   -- "Shaving..."
        #   :-- The senses...
        #   --J'aime l'ombre...
        #   -- Ce temps-là...
        #   - - J'aime...  if -- was split into two tokens
        .withColumn(
            "_is_single_dash_start",
            (
                (F.col("_is_dash_token") == 1)
                & (
                    F.col("_next_is_word")
                    | (F.col("_next_is_quote") & F.col("_next2_is_word"))
                )
            ),
        )
        .withColumn(
            "_is_split_double_hyphen_start",
            (
                (F.col("token") == "-")
                & (F.col("_next_token") == "-")
                & (
                    F.col("_next2_is_word")
                    | (F.col("_next2_is_quote") & F.col("_next3_is_word"))
                )
            ),
        )
        .withColumn(
            "_is_dash_dialog_start",
            F.when(
                (
                    F.col("_prev_is_boundary")
                    | F.col("_token_starts_colon_dash")
                )
                & (
                    F.col("_is_single_dash_start")
                    | F.col("_is_split_double_hyphen_start")
                    | F.col("_token_starts_dash_word")
                ),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "_is_sentence_end",
            F.when(F.col("token").isin(*sorted(_SENTENCE_END_TOKENS)), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "_dash_start_pos",
            F.when(F.col("_is_dash_dialog_start") == 1, F.col("_doc_pos"))
             .otherwise(F.lit(None).cast("long")),
        )
        .withColumn(
            "_sentence_end_pos",
            F.when(F.col("_is_sentence_end") == 1, F.col("_doc_pos"))
             .otherwise(F.lit(None).cast("long")),
        )
    )

    segmented = (
        labeled

        # Quote state, reset at chunk boundaries so one broken quote does not
        # contaminate a full document.
        .withColumn(
            "_directional_after",
            F.sum("_quote_delta").over(quote_window),
        )
        .withColumn(
            "_directional_before",
            F.col("_directional_after") - F.col("_quote_delta"),
        )
        .withColumn(
            "_toggle_after",
            F.sum("_is_toggle_quote").over(quote_window),
        )
        .withColumn(
            "_toggle_before",
            F.col("_toggle_after") - F.col("_is_toggle_quote"),
        )
        .withColumn(
            "_inside_directional_quote",
            (F.col("_directional_before") > 0) | (F.col("_directional_after") > 0),
        )
        .withColumn(
            "_inside_toggle_quote",
            (
                F.pmod(F.col("_toggle_before"), F.lit(2)) == 1
            ) | (
                F.pmod(F.col("_toggle_after"), F.lit(2)) == 1
            ),
        )

        # Dash-dialog state.
        # A dash-start opens a short dialogue span until the next sentence-final
        # punctuation. This avoids turning whole chapters into dialogue.
        .withColumn(
            "_last_dash_start_pos",
            F.coalesce(F.max("_dash_start_pos").over(dash_window), F.lit(-1)),
        )
        .withColumn(
            "_last_sentence_end_pos",
            F.coalesce(F.max("_sentence_end_pos").over(dash_window), F.lit(-1)),
        )
        .withColumn(
            "_inside_dash_dialog",
            F.col("_last_dash_start_pos") > F.col("_last_sentence_end_pos"),
        )

        .withColumn(
            "segment",
            F.when(
                F.col("_inside_directional_quote")
                | F.col("_inside_toggle_quote")
                | F.col("_inside_dash_dialog"),
                F.lit("dialog"),
            ).otherwise(F.lit("narration")),
        )
        .drop(
            "_doc_pos",
            "_prev_token",
            "_next_token",
            "_next2_token",
            "_next3_token",
            "_is_open_quote",
            "_is_close_quote",
            "_is_toggle_quote",
            "_quote_delta",
            "_is_dash_token",
            "_prev_is_boundary",
            "_next_is_word",
            "_next2_is_word",
            "_next3_is_word",
            "_next_is_quote",
            "_next2_is_quote",
            "_token_starts_dash",
            "_token_starts_dash_word",
            "_token_starts_colon_dash",
            "_is_single_dash_start",
            "_is_split_double_hyphen_start",
            "_is_dash_dialog_start",
            "_is_sentence_end",
            "_dash_start_pos",
            "_sentence_end_pos",
            "_directional_after",
            "_directional_before",
            "_toggle_after",
            "_toggle_before",
            "_inside_directional_quote",
            "_inside_toggle_quote",
            "_last_dash_start_pos",
            "_last_sentence_end_pos",
            "_inside_dash_dialog",
        )
    )

    return segmented


def compute_dialog_summary(segmented_df: DataFrame) -> DataFrame:
    """
    Aggregate dialog/narration token counts per document.

    Input: output of compute_dialog()

    Output columns:
        author, document_id, language,
        total_tokens, dialog_tokens, narration_tokens, dialog_ratio
    """
    _require_columns(
        segmented_df,
        {"author", "document_id", "language", "segment"},
        "compute_dialog_summary",
    )

    return (
        segmented_df
        .groupBy("author", "document_id", "language")
        .agg(
            F.count("*").alias("total_tokens"),
            F.sum(
                F.when(F.col("segment") == "dialog", F.lit(1)).otherwise(F.lit(0))
            ).alias("dialog_tokens"),
            F.sum(
                F.when(F.col("segment") == "narration", F.lit(1)).otherwise(F.lit(0))
            ).alias("narration_tokens"),
        )
        .withColumn(
            "dialog_ratio",
            (
                F.col("dialog_tokens")
                / F.greatest(F.col("total_tokens"), F.lit(1))
            ).cast("double"),
        )
        .orderBy("author", "document_id")
    )


# ---------------------------------------------------------------------------
# 3. Type-Token Ratio (TTR)
# ---------------------------------------------------------------------------

def compute_ttr(token_df: DataFrame) -> DataFrame:
    """
    Compute Type-Token Ratio per document.

    TTR = unique lemmas / total word tokens

    Uses lemma rather than raw token to normalize inflection.
    Excludes punctuation, symbols, and unknown tokens.

    Output columns:
        author, document_id, language,
        total_tokens, unique_lemmas, ttr
    """
    _require_columns(
        token_df,
        {"author", "document_id", "language", "lemma", "pos"},
        "compute_ttr",
    )

    words = token_df.filter(
        ~F.col("pos").isin("PUNCT", "SYM", "X")
    )

    return (
        words
        .groupBy("author", "document_id", "language")
        .agg(
            F.count("*").alias("total_tokens"),
            F.countDistinct(F.lower(F.col("lemma"))).alias("unique_lemmas"),
        )
        .withColumn(
            "ttr",
            (
                F.col("unique_lemmas")
                / F.greatest(F.col("total_tokens"), F.lit(1))
            ).cast("double"),
        )
        .orderBy("author", "document_id")
    )


# ---------------------------------------------------------------------------
# 4. POS distribution
# ---------------------------------------------------------------------------

_POS_OF_INTEREST = {
    "NOUN", "VERB", "ADJ", "PROPN", "ADV", "DET", "PUNCT",
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
}

# Normalise PTB tags to UD-style labels for cross-language comparison.
_PTB_TO_UD = {
    # Nom
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "PROPN",
    "NNPS": "PROPN",
    # Verbes
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    # Adjectifs
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    # Adverbes
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    # Determinants
    "DT": "DET",
    "PDT": "DET",
    "WDT": "DET",
    # Ponctuation
    ".": "PUNCT",
    ",": "PUNCT",
    ":": "PUNCT",
    ";": "PUNCT",
    "''": "PUNCT",
    "``": "PUNCT",
    "-LRB-": "PUNCT",
    "-RRB-": "PUNCT",
    "HYPH": "PUNCT",
    "NFP": "PUNCT",
}


def _normalise_pos_udf():
    """
    Return a small UDF that maps English PTB tags to UD-style labels.

    French Spark NLP already uses UD-style POS tags. English Spark NLP uses
    Penn Treebank tags, so we normalize only the PTB subset needed for plots.
    """
    mapping = dict(_PTB_TO_UD)

    def _norm(pos):
        return mapping.get(pos, pos)

    return F.udf(_norm, T.StringType())


_norm_pos_udf = _normalise_pos_udf()


def compute_pos_distribution(token_df: DataFrame) -> DataFrame:
    """
    Compute per-document POS distribution.

    PTB tags (EN) are normalised to UD-style labels so FR and EN are
    comparable in plots.

    Output columns:
        author, document_id, language, pos_normalised,
        token_count, total_tokens, pos_ratio
    """
    _require_columns(
        token_df,
        {"author", "document_id", "language", "pos"},
        "compute_pos_distribution",
    )

    normed = (
        token_df
        .withColumn("pos_normalised", _norm_pos_udf(F.col("pos")))
        .filter(
            F.col("pos_normalised").isin(
                "NOUN", "VERB", "ADJ", "PROPN", "ADV", "DET", "PUNCT"
            )
        )
    )

    per_doc_total = (
        token_df
        .groupBy("author", "document_id", "language")
        .agg(F.count("*").alias("total_tokens"))
    )

    per_pos = (
        normed
        .groupBy("author", "document_id", "language", "pos_normalised")
        .agg(F.count("*").alias("token_count"))
    )

    return (
        per_pos
        .join(per_doc_total, on=["author", "document_id", "language"], how="left")
        .withColumn(
            "pos_ratio",
            (
                F.col("token_count")
                / F.greatest(F.col("total_tokens"), F.lit(1))
            ).cast("double"),
        )
        .orderBy("author", "document_id", "pos_normalised")
    )


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_stylometry(df: DataFrame, output_dir: str) -> None:
    """
    Save a stylometry DataFrame partitioned by author.

    The output path is overwritten intentionally because each run should
    represent one consistent version of the pipeline.
    """
    (
        df
        .write
        .mode("overwrite")
        .partitionBy("author")
        .parquet(output_dir)
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_and_save_stylometry(
    spark: SparkSession,
    annotated_dir: str = "DATA/processed/corpus_annotated",
    zipf_out: str = "DATA/processed/stylometry_zipf",
    dialog_out: str = "DATA/processed/stylometry_dialog",
    ttr_out: str = "DATA/processed/stylometry_ttr",
    pos_out: str = "DATA/processed/stylometry_pos",
) -> None:
    """
    Full stylometry pipeline:
      1. Load annotated corpus
      2. Compute Zipf rank-frequency data
      3. Compute dialog/narration segmentation summary
      4. Compute type-token ratio
      5. Compute POS distribution
      6. Save all four outputs to Parquet
    """
    print("[stylometry] Loading annotated corpus...")
    token_df = spark.read.parquet(annotated_dir).cache()

    try:
        print(f"[stylometry] Token rows: {token_df.count()}")

        print("[stylometry] Computing Zipf data...")
        zipf_df = compute_zipf(token_df)
        save_stylometry(zipf_df, zipf_out)
        print(f"[stylometry] Zipf saved to {zipf_out}")

        print("[stylometry] Computing dialog/narration segmentation...")
        segmented_df = compute_dialog(token_df)
        dialog_summary = compute_dialog_summary(segmented_df)
        dialog_summary.show(20, truncate=False)
        save_stylometry(dialog_summary, dialog_out)
        print(f"[stylometry] Dialog summary saved to {dialog_out}")

        print("[stylometry] Computing type-token ratio...")
        ttr_df = compute_ttr(token_df)
        ttr_df.show(20, truncate=False)
        save_stylometry(ttr_df, ttr_out)
        print(f"[stylometry] TTR saved to {ttr_out}")

        print("[stylometry] Computing POS distribution...")
        pos_df = compute_pos_distribution(token_df)
        save_stylometry(pos_df, pos_out)
        print(f"[stylometry] POS distribution saved to {pos_out}")

    finally:
        token_df.unpersist()

    print("[stylometry] Done.")