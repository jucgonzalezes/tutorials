import polars as pl

from tutorials.utils.utils import new_section

if __name__ == "__main__":
    df = pl.read_parquet(
        "tutorials/datasets/green_tripdata_2024/green_tripdata_2024-01.parquet"
    )

    # Head
    new_section("Head")
    print(df.head(5))

    # Shape
    new_section("Shape")
    print(df.shape)

    # Columns
    new_section("DataFrame Columns")
    print(df.columns)

    # Describe
    new_section("Describe df")
    print(df.describe())

    # Schema
    new_section("Collect Schema")
    print(df.collect_schema())

    # Unique values per column
    new_section("Unique")
    col = "RatecodeID"
    print(df.select(col).unique())

    # Groupby + aggregate
    new_section("Groupby + Aggregate")
    print("Description:")
    print(
        "Groups by RatecodeID, aggregares by mean total amount, mean tip amount, trip count,\n\
          and total passenger sum. Then filters by RatecodeID excluding null codes.\n"
    )
    print(
        df.group_by(col)
        .agg(
            [
                pl.col("total_amount").mean().alias("Mean amount"),
                pl.col("tip_amount").mean().alias("Mean tip"),
                pl.len().alias("Total trips"),
                pl.col("passenger_count").sum().alias("Total passenger count"),
            ]
        )
        .sort(by="Mean amount", descending=True)
        .filter(pl.col("RatecodeID").is_not_null())
    )

    # Window Aggregation
    new_section("Window Aggregation")
    print("#TODO")
