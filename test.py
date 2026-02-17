# Make sure these are already defined:
# legacy_df = spark.read.parquet("your/path")
# delta_df = apply_transformations(legacy_df)  # your transformation function

# Then run this in a new cell:
joined_test = legacy_df.alias("legacy").join(delta_df.alias("delta"), on="ACCT_ID", how="inner").limit(100)

# Test case 1b (simplest)
try:
    joined_test.withColumn(
        "case1b_valid",
        F.isnull(F.col("delta.assets")) | (F.size(F.col("delta.assets")) == 0)
    ).first()
    print("Case 1b expression: OK")
except Exception as e:
    print(f"Case 1b expression FAILS: {str(e)[:100]}")

# Test case 2 (medium complexity)
try:
    joined_test.withColumn(
        "case2_valid",
        F.expr("array_sort(transform(`legacy`.`assets`, x -> to_json(x)))") == 
        F.expr("array_sort(transform(`delta`.`assets`, x -> to_json(x)))")
    ).first()
    print("Case 2 expression: OK")
except Exception as e:
    print(f"Case 2 expression FAILS: {str(e)[:100]}")

# Test case 1a (most complex)
try:
    joined_test.withColumn(
        "case1a_valid",
        F.expr("array_sort(transform(filter(`legacy`.`INCOME`, x -> x.ASSETS_AMOUNT is not null), x -> to_json(x)))")
    ).first()
    print("Case 1a expression: OK")
except Exception as e:
    print(f"Case 1a expression FAILS: {str(e)[:100]}")
