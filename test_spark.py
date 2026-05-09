from src.spark_session import create_spark_session

spark = create_spark_session()

df = spark.createDataFrame(
    [
        ("Balzac", "Le Père Goriot"),
        ("Flaubert", "Madame Bovary"),
        ("Zola", "Germinal"),
    ],
    ["author", "title"]
)

df.show()

spark.stop()