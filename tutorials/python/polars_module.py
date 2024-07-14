import polars as pl

if __name__ =="__main__": 
    df = pl.read_parquet("tutorials/datasets/green_tripdata_2024/green_tripdata_2024-01.parquet")
    print(df.columns)
    print(df.head(5))
    print(df.describe())
    print(df.collect_schema())