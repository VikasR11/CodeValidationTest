from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType
from pyspark import StorageLevel
from typing import Dict
import time


def compare_legacy_vs_delta(
    legacy_df: DataFrame,
    delta_df: DataFrame,
    primary_key: str,
    num_partitions: int = 0
) -> Dict:
    """
    Validate delta matches legacy with order-insensitive array/struct comparison.
    
    Args:
        legacy_df: Source DataFrame
        delta_df: Target DataFrame  
        primary_key: Primary key column
        num_partitions: Repartition count (0 = skip, use 300-400 for large data)
    
    Returns:
        Dict with validation results
    """
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"VALIDATION - Order-Insensitive Comparison")
    print(f"{'='*70}\n")
    
    # Mappings
    income_to_assets = {
        'INCM_SOURCE': 'assets_source',
        'INCOME_CHANNEL_NAME': 'assets_channel_name',
        'STRATEGY_NAME': 'assets_strategy_name',
        'INCOME_LAST_MODIFIED_SOURCE_SYSTEM': 'assets_last_modified_source_system',
        'INCOME_REPORTED_TS': 'assets_reported_utc_timestamp',
        'ASSETS_AMOUNT': 'assets_amount',
        'UNTAXED_INCOME_AMOUNT': 'untaxed_income_amount',
    }
    
    excluded = {'assets', 'INCOME'}
    shared_cols = list((set(legacy_df.columns) & set(delta_df.columns)) - {primary_key} - excluded)
    has_assets = "assets" in legacy_df.columns
    
    results = {
        "step1_pk_validation": {},
        "step2_content_validation": {},
        "step3_assets_income_validation": {},
        "overall_validation_passed": True,
        "execution_time_seconds": 0
    }
    
    # Optional repartition
    if num_partitions > 0:
        print(f"Repartitioning to {num_partitions} partitions...")
        leg = legacy_df.repartition(num_partitions, primary_key)
        dlt = delta_df.repartition(num_partitions, primary_key)
        leg.persist(StorageLevel.MEMORY_AND_DISK)
        dlt.persist(StorageLevel.MEMORY_AND_DISK)
    else:
        print("Using original partitioning")
        leg = legacy_df
        dlt = delta_df
    
    # ========== STEP 1: PK VALIDATION ==========
    print("\nStep 1: PK Validation...")
    
    pk_outer = leg.select(primary_key).alias("l").join(
        dlt.select(primary_key).alias("d"),
        on=primary_key,
        how="outer"
    ).select(
        F.when(F.col(f"l.{primary_key}").isNull(), F.lit("extra"))
         .when(F.col(f"d.{primary_key}").isNull(), F.lit("missing"))
         .otherwise(F.lit("ok")).alias("status")
    )
    
    pk_fail = pk_outer.filter(F.col("status") != "ok").first()
    
    results["step1_pk_validation"] = {"passed": pk_fail is None}
    
    if pk_fail:
        print("Step 1 FAILED")
        results["overall_validation_passed"] = False
        results["execution_time_seconds"] = time.time() - start_time
        if num_partitions > 0:
            leg.unpersist()
            dlt.unpersist()
        return results
    
    print("Step 1 PASSED")
    
    # ========== STEP 2: CONTENT VALIDATION ==========
    print("\nStep 2: Content Validation...")
    
    if not shared_cols:
        print("No shared columns")
        results["step2_content_validation"] = {"passed": True}
    else:
        joined = leg.alias("legacy").join(dlt.alias("delta"), on=primary_key, how="inner")
        
        # Build comparison for each column type
        comparisons = []
        
        # Get schema once
        schema_map = {f.name: f.dataType for f in legacy_df.schema.fields}
        
        for col in shared_cols:
            col_type = schema_map[col]
            
            if isinstance(col_type, ArrayType) and isinstance(col_type.elementType, StructType):
                # Array of structs - sort fields, serialize, sort array
                if col == "HOUSING":
                    fields = sorted([f.name for f in col_type.elementType.fields])
                else:
                    fields = sorted([f.name for f in col_type.elementType.fields])
                
                fs = ", ".join([f"x.`{f}` as `{f}`" for f in fields])
                comp = (
                    F.expr(f"array_sort(transform(`legacy`.`{col}`, x -> to_json(struct({fs}))))") ==
                    F.expr(f"array_sort(transform(`delta`.`{col}`, x -> to_json(struct({fs}))))")
                )
                comparisons.append(comp | (F.col(f"legacy.{col}").isNull() & F.col(f"delta.{col}").isNull()))
                
            elif isinstance(col_type, ArrayType):
                # Simple array - just sort
                comparisons.append(
                    (F.array_sort(F.col(f"legacy.{col}")) == F.array_sort(F.col(f"delta.{col}"))) |
                    (F.col(f"legacy.{col}").isNull() & F.col(f"delta.{col}").isNull())
                )
                
            elif isinstance(col_type, StructType):
                # Struct - sort fields, serialize
                fields = sorted([f.name for f in col_type.fields])
                leg_fs = ", ".join([f"`legacy`.`{col}`.`{f}` as `{f}`" for f in fields])
                dlt_fs = ", ".join([f"`delta`.`{col}`.`{f}` as `{f}`" for f in fields])
                comparisons.append(
                    (F.expr(f"to_json(struct({leg_fs}))") == F.expr(f"to_json(struct({dlt_fs}))")) |
                    (F.col(f"legacy.{col}").isNull() & F.col(f"delta.{col}").isNull())
                )
                
            else:
                # Primitive - direct comparison
                comparisons.append(
                    (F.col(f"legacy.{col}") == F.col(f"delta.{col}")) |
                    (F.col(f"legacy.{col}").isNull() & F.col(f"delta.{col}").isNull())
                )
        
        # Combine all comparisons
        if comparisons:
            all_match = comparisons[0]
            for c in comparisons[1:]:
                all_match = all_match & c
            
            mismatch = joined.filter(~all_match).first()
            
            results["step2_content_validation"] = {"passed": mismatch is None}
            
            if mismatch:
                print("Step 2 FAILED")
                results["overall_validation_passed"] = False
                results["execution_time_seconds"] = time.time() - start_time
                if num_partitions > 0:
                    leg.unpersist()
                    dlt.unpersist()
                return results
        else:
            results["step2_content_validation"] = {"passed": True}
        
        print("Step 2 PASSED")
    
    # ========== STEP 3: ASSETS/INCOME VALIDATION ==========
    print("\nStep 3: Assets/INCOME Validation...")
    
    # Narrow to only columns needed for Step 3 to reduce memory pressure
    # Keep: PK, legacy.INCOME, legacy.assets (if exists), delta.assets
    legacy_step3_cols = [primary_key, "INCOME"] + (["assets"] if has_assets else [])
    delta_step3_cols = [primary_key, "assets"]
    
    leg_narrow = leg.select(*legacy_step3_cols)
    dlt_narrow = dlt.select(*delta_step3_cols)
    
    # Join narrow DataFrames
    joined = leg_narrow.alias("legacy").join(dlt_narrow.alias("delta"), on=primary_key, how="inner")
    joined = joined.coalesce(200).persist(StorageLevel.MEMORY_AND_DISK)
    
    # Get INCOME fields
    income_fields = []
    for f in legacy_df.schema.fields:
        if f.name == "INCOME" and isinstance(f.dataType, ArrayType):
            income_fields = [sf.name for sf in f.dataType.elementType.fields]
            break
    
    # Build qualifying condition
    # Qualifying = (ASSETS_AMOUNT not null OR UNTAXED_INCOME_AMOUNT not null) AND INCM_SOURCE == "CUSTOMER_PROVIDED"
    amount_quals = []
    if "ASSETS_AMOUNT" in income_fields:
        amount_quals.append("x.ASSETS_AMOUNT is not null")
    if "UNTAXED_INCOME_AMOUNT" in income_fields:
        amount_quals.append("x.UNTAXED_INCOME_AMOUNT is not null")
    
    if amount_quals and "INCM_SOURCE" in income_fields:
        # Both amount fields check AND source check
        amount_expr = " or ".join(amount_quals)
        qual_expr = f"({amount_expr}) and x.INCM_SOURCE == 'CUSTOMER_PROVIDED'"
    elif amount_quals:
        # Only amount fields exist, no source check
        qual_expr = " or ".join(amount_quals)
    else:
        # No qualifying fields exist
        qual_expr = "false"
    
    # Categorize
    if has_assets:
        cat = joined.withColumn(
            "leg_valid",
            ~(F.isnull(F.col("legacy.assets")) | (F.size(F.col("legacy.assets")) == 0))
        ).withColumn(
            "has_qual",
            F.expr(f"exists(`legacy`.`INCOME`, x -> {qual_expr})")
        )
    else:
        cat = joined.withColumn("leg_valid", F.lit(False)) \
                    .withColumn("has_qual", F.expr(f"exists(`legacy`.`INCOME`, x -> {qual_expr})"))
    
    # Compute all case validations in a SINGLE pass with one .first() call
    # This avoids multiple expensive .first() calls on complex expressions
    
    exp_fields = sorted(income_to_assets.values())
    l2d = {v: k for k, v in income_to_assets.items()}
    mapped = ", ".join([f"x.`{l2d[d]}` as `{d}`" for d in exp_fields])
    dlt_norm = ", ".join([f"x.`{f}` as `{f}`" for f in exp_fields])
    
    asset_fields = None
    if has_assets:
        for f in legacy_df.schema.fields:
            if f.name == "assets" and isinstance(f.dataType, ArrayType):
                asset_fields = sorted([sf.name for sf in f.dataType.elementType.fields])
                break
    
    # Build all case columns at once
    step3_df = cat
    
    # Case 1b: delta.assets should be null/empty
    step3_df = step3_df.withColumn(
        "case1b_valid",
        F.when(
            (~F.col("leg_valid")) & (~F.col("has_qual")),
            F.isnull(F.col("delta.assets")) | (F.size(F.col("delta.assets")) == 0)
        ).otherwise(F.lit(True))
    )
    
    # Case 1a: delta.assets should match mapped INCOME fields
    # Only build this expression if qualifying fields actually exist in INCOME schema
    if amount_quals:
        step3_df = step3_df.withColumn(
            "case1a_valid",
            F.when(
                (~F.col("leg_valid")) & F.col("has_qual"),
                # If delta.assets is null when we expect mapped values, that's a failure
                F.when(
                    F.col("delta.assets").isNull(),
                    F.lit(False)
                ).otherwise(
                    F.expr(f"array_sort(transform(filter(`legacy`.`INCOME`, x -> {qual_expr}), x -> to_json(struct({mapped}))))") ==
                    F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({dlt_norm}))))")
                )
            ).otherwise(F.lit(True))
        )
    else:
        # ASSETS_AMOUNT and UNTAXED_INCOME_AMOUNT don't exist in INCOME schema
        # No rows can qualify for Case 1a - skip entirely
        step3_df = step3_df.withColumn("case1a_valid", F.lit(True))
    
    # Case 2: delta.assets should match legacy.assets
    if asset_fields:
        fs = ", ".join([f"x.`{f}` as `{f}`" for f in asset_fields])
        step3_df = step3_df.withColumn(
            "case2_valid",
            F.when(
                F.col("leg_valid"),
                # Handle null comparisons explicitly - coalesce to False if comparison is null
                F.coalesce(
                    (F.expr(f"array_sort(transform(`legacy`.`assets`, x -> to_json(struct({fs}))))") ==
                     F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({fs}))))")) |
                    (F.col("legacy.assets").isNull() & F.col("delta.assets").isNull()),
                    F.lit(False)
                )
            ).otherwise(F.lit(True))
        )
    else:
        step3_df = step3_df.withColumn("case2_valid", F.lit(True))
    
    # Single combined column - any row failing ANY case
    step3_df = step3_df.withColumn(
        "step3_valid",
        F.col("case1a_valid") & F.col("case1b_valid") & F.col("case2_valid")
    )
    
    # ONE .first() call to find any failure
    try:
        step3_fail = step3_df.filter(~F.col("step3_valid")).first()
        
        if step3_fail is None:
            case1a_pass = True
            case1b_pass = True
            case2_pass = True
        else:
            row = step3_fail.asDict()
            case1a_pass = bool(row.get("case1a_valid", True))
            case1b_pass = bool(row.get("case1b_valid", True))
            case2_pass = bool(row.get("case2_valid", True))
            print(f"  Failing row PK: {row.get(primary_key)}")
    except Exception as e:
        print(f"  ERROR: Step 3 validation failed to execute: {str(e)[:150]}")
        case1a_pass = None
        case1b_pass = None
        case2_pass = None
    
    step3_pass = None if case1a_pass is None else (case1a_pass and case1b_pass and case2_pass)
    
    results["step3_assets_income_validation"] = {
        "passed": step3_pass,
        "case1a_passed": case1a_pass,
        "case1b_passed": case1b_pass,
        "case2_passed": case2_pass
    }
    
    if step3_pass is None:
        print("Step 3 INCOMPLETE - Could not execute validation")
        results["overall_validation_passed"] = False
    elif not step3_pass:
        print("Step 3 FAILED")
        print(f"  Case 1a: {'PASS' if case1a_pass else 'FAIL'}")
        print(f"  Case 1b: {'PASS' if case1b_pass else 'FAIL'}")
        print(f"  Case 2: {'PASS' if case2_pass else 'FAIL'}")
        results["overall_validation_passed"] = False
    else:
        print("Step 3 PASSED")
    
    # Cleanup
    joined.unpersist()
    if num_partitions > 0:
        leg.unpersist()
        dlt.unpersist()
    
    results["execution_time_seconds"] = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"{'PASSED' if results['overall_validation_passed'] else 'FAILED'} ({results['execution_time_seconds']:.1f}s)")
    print(f"{'='*70}\n")
    
    return results
