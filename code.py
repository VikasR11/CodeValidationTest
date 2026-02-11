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
        print("❌ Step 1 FAILED")
        results["overall_validation_passed"] = False
        results["execution_time_seconds"] = time.time() - start_time
        if num_partitions > 0:
            leg.unpersist()
            dlt.unpersist()
        return results
    
    print("✓ Step 1 PASSED")
    
    # ========== STEP 2: CONTENT VALIDATION ==========
    print("\nStep 2: Content Validation...")
    
    if not shared_cols:
        print("No shared columns")
        results["step2_content_validation"] = {"passed": True}
        joined = None
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
                    F.expr(f"array_sort(transform(`delta`.`{col}`, x -> to_json(struct({fs}))))"))
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
                print("❌ Step 2 FAILED")
                results["overall_validation_passed"] = False
                results["execution_time_seconds"] = time.time() - start_time
                if num_partitions > 0:
                    leg.unpersist()
                    dlt.unpersist()
                return results
        else:
            results["step2_content_validation"] = {"passed": True}
        
        print("✓ Step 2 PASSED")
    
    # ========== STEP 3: ASSETS/INCOME VALIDATION ==========
    print("\nStep 3: Assets/INCOME Validation...")
    
    # Create or reuse join
    if joined is None:
        joined = leg.alias("legacy").join(dlt.alias("delta"), on=primary_key, how="inner")
    
    joined = joined.coalesce(200).persist(StorageLevel.MEMORY_AND_DISK)
    
    # Get INCOME fields
    income_fields = []
    for f in legacy_df.schema.fields:
        if f.name == "INCOME" and isinstance(f.dataType, ArrayType):
            income_fields = [sf.name for sf in f.dataType.elementType.fields]
            break
    
    # Build qualifying condition
    quals = []
    if "ASSETS_AMOUNT" in income_fields:
        quals.append("x.ASSETS_AMOUNT is not null")
    if "UNTAXED_INCOME_AMOUNT" in income_fields:
        quals.append("x.UNTAXED_INCOME_AMOUNT is not null")
    qual_expr = " or ".join(quals) if quals else "false"
    
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
    
    # Case 2: legacy.assets valid
    case2_pass = True
    if has_assets and cat.filter(F.col("leg_valid")).first():
        # Get asset fields
        asset_fields = None
        for f in legacy_df.schema.fields:
            if f.name == "assets" and isinstance(f.dataType, ArrayType):
                asset_fields = sorted([sf.name for sf in f.dataType.elementType.fields])
                break
        
        if asset_fields:
            fs = ", ".join([f"x.`{f}` as `{f}`" for f in asset_fields])
            c2 = cat.filter(F.col("leg_valid")).withColumn(
                "match",
                (F.expr(f"array_sort(transform(`legacy`.`assets`, x -> to_json(struct({fs}))))") ==
                 F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({fs}))))")) |
                (F.col("legacy.assets").isNull() & F.col("delta.assets").isNull())
            )
            case2_pass = c2.filter(~F.col("match")).first() is None
    
    # Case 1a: legacy.assets invalid, has qualifying income
    case1a_pass = True
    c1a_filter = cat.filter((~F.col("leg_valid")) & F.col("has_qual"))
    if c1a_filter.first():
        exp_fields = sorted(income_to_assets.values())
        l2d = {v: k for k, v in income_to_assets.items()}
        mapped = ", ".join([f"x.`{l2d[d]}` as `{d}`" for d in exp_fields])
        dlt_norm = ", ".join([f"x.`{f}` as `{f}`" for f in exp_fields])
        
        c1a = c1a_filter.withColumn(
            "exp",
            F.expr(f"array_sort(transform(filter(`legacy`.`INCOME`, x -> {qual_expr}), x -> to_json(struct({mapped}))))")
        ).withColumn(
            "dlt_norm",
            F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({dlt_norm}))))")
        )
        case1a_pass = c1a.filter(F.col("exp") != F.col("dlt_norm")).first() is None
    
    # Case 1b: legacy.assets invalid, no qualifying income
    case1b_pass = True
    c1b_filter = cat.filter((~F.col("leg_valid")) & (~F.col("has_qual")))
    if c1b_filter.first():
        c1b = c1b_filter.withColumn(
            "null_empty",
            F.isnull(F.col("delta.assets")) | (F.size(F.col("delta.assets")) == 0)
        )
        case1b_pass = c1b.filter(~F.col("null_empty")).first() is None
    
    step3_pass = case1a_pass and case1b_pass and case2_pass
    
    results["step3_assets_income_validation"] = {
        "passed": step3_pass,
        "case1a_passed": case1a_pass,
        "case1b_passed": case1b_pass,
        "case2_passed": case2_pass
    }
    
    if not step3_pass:
        print("❌ Step 3 FAILED")
        print(f"  Case 1a: {'PASS' if case1a_pass else 'FAIL'}")
        print(f"  Case 1b: {'PASS' if case1b_pass else 'FAIL'}")
        print(f"  Case 2: {'PASS' if case2_pass else 'FAIL'}")
        results["overall_validation_passed"] = False
    else:
        print("✓ Step 3 PASSED")
    
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
