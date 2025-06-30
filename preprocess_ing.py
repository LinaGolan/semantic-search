import polars as pl

# Load from Parquet URL directly (or you can download first)
df = pl.read_parquet("food.parquet")

# Query
df_filtered = df.select(["product_name", "brands", "countries_tags", "nutriscore_grade"]).filter(pl.col("nutriscore_grade").is_not_null())

print(df_filtered.head())
