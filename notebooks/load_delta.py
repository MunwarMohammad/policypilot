# Databricks notebook source
"""Upload data/chunks.jsonl into a UC Volume, then run this notebook.

The export must contain the COMPLETE active snapshot for each included company.
A per-company Delta transaction replaces those rows, avoiding duplicates and stale chunks.
Existing companies not present in the export remain unchanged.
"""

import re

# Databricks provides spark and dbutils in the notebook runtime.
from databricks.sdk.runtime import dbutils, spark
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

dbutils.widgets.text("table", "policypilot_dev.filings.chunks")
dbutils.widgets.text("input_path", "/Volumes/policypilot_dev/filings/raw/chunks.jsonl")
table = dbutils.widgets.get("table")
path = dbutils.widgets.get("input_path")
if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*", table):
    raise ValueError("Expected a three-part catalog.schema.table identifier")
fields = [
    StructField(k, StringType(), False)
    for k in (
        "ticker",
        "company",
        "accession",
        "filing_date",
        "report_date",
        "source_url",
        "chunk_id",
        "text",
        "embedding_model",
    )
]
schema = StructType(
    fields
    + [
        StructField("cik", LongType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("embedding", ArrayType(DoubleType()), False),
    ]
)
df = spark.read.schema(schema).json(path).cache()
if df.count() == 0:
    raise ValueError("Export is empty")
if df.select("embedding_model").distinct().count() != 1:
    raise ValueError("Export mixes embedding models")
if df.where(F.size("embedding") != 384).count():
    raise ValueError("Expected 384-dimensional MiniLM embeddings")
if df.select("chunk_id").distinct().count() != df.count():
    raise ValueError("Duplicate chunk IDs")
if df.where(F.col("ticker").isNull() | F.col("text").isNull() | F.col("chunk_id").isNull()).count():
    raise ValueError("Required fields are missing")
(
    df.limit(0)
    .write.format("delta")
    .mode("ignore")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(table)
)
spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
for row in df.select("ticker").distinct().collect():
    ticker = row.ticker
    if not re.fullmatch(r"[A-Z]{1,6}", ticker):
        raise ValueError("Invalid ticker")
    (
        df.where(F.col("ticker") == ticker)
        .write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"ticker = '{ticker}'")
        .saveAsTable(table)
    )
df.unpersist()
