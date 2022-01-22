import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

CSV_PATH = "data.csv"

spark = (
    SparkSession.builder.
    master("local[1]").
    appName("tutorial").
    getOrCreate()
)


def make_data():
    data = [
        ("James", "Sales", "NY", 90000, 34, 10000),
        ("Michael", "Sales", "NY", 86000, 56, 20000),
        ("Robert", "Sales", "CA", 81000, 30, 23000),
        ("Maria", "Finance", "CA", 90000, 24, 23000),
        ("Raman", "Finance", "CA", 99000, 40, 24000),
        ("Scott", "Finance", "NY", 83000, 36, 19000),
        ("Jen", "Finance", "NY", 79000, 53, 15000),
        ("Jeff", "Marketing", "CA", 80000, 25, 18000),
        ("Kumar", "Marketing", "NY", 91000, 50, 21000)
    ]

    schema = ["employee_name", "department", "state", "salary", "age", "bonus"]
    df = pd.DataFrame(data, columns=schema)
    df.to_csv(CSV_PATH, index=False)


def main():
    make_data()

    # Read data
    df = (
        spark.read.
        options(header=True, delimiter=",", inferSchema=True).
        csv(CSV_PATH)
    )
    df.show(5)

    # Create a column
    df = df.withColumn("compensation", F.col("salary") + F.col("bonus"))

    # Grouping
    agg = [F.mean(c).alias(c) for c in ("compensation", "salary", "bonus")]
    state_summary = df.groupBy(["state"]).agg(*agg)
    state_summary.show(5)

    # Concat axis 0
    concat = df.union(df)
    concat.show(5)

    # Merge
    keys = ["employee_name", "department", "state"]
    df_left = df
    df_right = df
    for c in df.columns:
        if c not in keys:
            df_left = df_left.withColumnRenamed(c, f"{c}_left")
            df_right = df_right.withColumnRenamed(c, f"{c}_right")
    merged = df_left.join(df_right, keys)
    merged.show(5)
