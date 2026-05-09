from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType


LANGUAGE_MAP = {
    "balzac": "fr",
    "dumas": "fr",
    "stendhal": "fr",
    "hugo": "fr",
    "sand": "en",      # Current Indiana file is the English translation.
    "flaubert": "fr",
    "zola": "fr",
    "dickens": "en",
}


ERA_MAP = {
    "balzac": "romantic",
    "dumas": "romantic",
    "stendhal": "romantic",
    "hugo": "romantic",
    "sand": "romantic",
    "flaubert": "realist",
    "zola": "realist",
    "dickens": "realist",
}


SCHEMA = StructType(
    [
        StructField("raw_text", StringType(), False),
        StructField("source_path", StringType(), False),
        StructField("author", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("language", StringType(), False),
        StructField("era", StringType(), False),
    ]
)


def load_text_corpus(spark: SparkSession, data_dir: str = "DATA/raw") -> DataFrame:
    """
    Load all .txt files from DATA/raw into a Spark DataFrame.

    Expected layout:
        DATA/raw/{author}/{title}.txt

    pathlib is used intentionally because it is reliable on Windows.
    """
    base_dir = Path(data_dir)
    txt_files = sorted(base_dir.glob("*/*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")

    rows = []

    for path in txt_files:
        author = path.parent.name.lower()
        filename = path.name
        document_id = path.stem
        source_path = str(path.resolve())
        raw_text = path.read_text(encoding="utf-8")

        if author not in LANGUAGE_MAP:
            raise ValueError(f"Missing language mapping for author: {author}")

        if author not in ERA_MAP:
            raise ValueError(f"Missing era mapping for author: {author}")

        rows.append(
            {
                "raw_text": raw_text,
                "source_path": source_path,
                "author": author,
                "filename": filename,
                "document_id": document_id,
                "language": LANGUAGE_MAP[author],
                "era": ERA_MAP[author],
            }
        )

    return spark.createDataFrame(rows, schema=SCHEMA)


def save_corpus(df: DataFrame, output_dir: str, fmt: str = "parquet") -> None:
    """
    Save corpus as Parquet or ORC, partitioned by author.

    Unlike the older version, this writes exactly to output_dir.
    It does not append _parquet or _orc automatically.
    """
    fmt = fmt.lower()

    if fmt not in {"parquet", "orc"}:
        raise ValueError("fmt must be 'parquet' or 'orc'")

    writer = (
        df.write
        .mode("overwrite")
        .partitionBy("author")
    )

    if fmt == "parquet":
        writer.parquet(output_dir)
    else:
        writer.orc(output_dir)