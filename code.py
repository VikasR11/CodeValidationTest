from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType
from pyspark import StorageLevel
from typing import Dict, List
import time


def compare_legacy_vs_delta(legacy_df: DataFrame, delta_df: DataFrame, primary_key: str, num_partitions: int = 300, materialize_inputs: bool = True) -> Dict:
    """
    Validate that delta DataFrame matches legacy DataFrame (source of truth) for 230M rows.
    
    Validation Steps:
    1. PK Validation: Ensure same set of primary keys in both DataFrames
    2. Content Validation: Compare shared columns (excluding assets and INCOME)
    3. Assets/INCOME Validation:
       - Case 1: legacy.assets is null/empty/missing
         - Case 1a: If qualifying INCOME structs exist (non-null ASSETS_AMOUNT/UNTAXED_INCOME_AMOUNT),
                    delta.assets should contain mapped values
         - Case 1b: If no qualifying INCOME structs OR if ASSETS_AMOUNT/UNTAXED_INCOME_AMOUNT fields
                    don't exist in schema, delta.assets should be null
       - Case 2: legacy.assets is valid → delta.assets should match exactly (order-insensitive)
    
    Args:
        legacy_df: Source of truth DataFrame in Parquet format
        delta_df: Modern DataFrame in Delta Lake format (or transformed parquet)
        primary_key: Single column name to use as primary key
        num_partitions: Number of partitions to repartition to (default: 300). 
                       Set to 0 to skip repartitioning and use original partitioning.
                       Try 300-400 for 200-core cluster with complex nested data.
        materialize_inputs: Cache and materialize inputs to avoid lineage issues (default: True)
    
    Returns:
        Dictionary with validation results for all steps
    """
    
    start_time = time.time()
    
    # Materialize inputs to avoid lineage explosion issues
    # This is critical when delta_df is a transformed parquet with long lineage
    if materialize_inputs:
        print("Materializing input DataFrames (lazy - will materialize on first operation)...")
        legacy_df.persist(StorageLevel.MEMORY_AND_DISK)
        delta_df.persist(StorageLevel.MEMORY_AND_DISK)
        print("Persist marked - caching will occur on first use")
    else:
        # Still cache for efficiency
        legacy_df.persist(StorageLevel.MEMORY_AND_DISK)
        delta_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Define excluded columns for step 2 (handled in step 3)
    excluded_columns = ['assets', 'INCOME']
    
    # Define income to assets field mapping for Case 1a
    # Maps legacy.INCOME fields to delta.assets fields
    income_to_assets_mapping = {
        'INCM_SOURCE': 'assets_source',
        'INCOME_CHANNEL_NAME': 'assets_channel_name',
        'STRATEGY_NAME': 'assets_strategy_name',
        'INCOME_LAST_MODIFIED_SOURCE_SYSTEM': 'assets_last_modified_source_system',
        'INCOME_REPORTED_TS': 'assets_reported_utc_timestamp',
        'ASSETS_AMOUNT': 'assets_amount',
        'UNTAXED_INCOME_AMOUNT': 'untaxed_income_amount',
    }
    
    results = {
        "step1_pk_validation": {},
        "step2_content_validation": {},
        "step3_assets_income_validation": {},
        "overall_validation_passed": True,
        "execution_time_seconds": 0
    }
    
    # Optionally repartition DataFrames for better join performance
    if num_partitions > 0:
        print(f"Repartitioning DataFrames into {num_partitions} partitions...")
        legacy_repartitioned = legacy_df.repartition(num_partitions, primary_key)
        delta_repartitioned = delta_df.repartition(num_partitions, primary_key)
    else:
        print(f"Using original partitioning (legacy: {legacy_df.rdd.getNumPartitions()}, delta: {delta_df.rdd.getNumPartitions()})")
        legacy_repartitioned = legacy_df
        delta_repartitioned = delta_df
    
    # ========================================
    # STEP 1: Primary Key Validation (Simplified approach)
    # ========================================
    print("Starting Step 1: PK Validation...")
    
    # Check if there are PKs in legacy not in delta (missing in delta)
    print("Checking for PKs missing in delta...")
    missing_in_delta = legacy_repartitioned.select(primary_key).subtract(
        delta_repartitioned.select(primary_key)
    ).first()
    has_missing = (missing_in_delta is not None)
    
    # Check if there are PKs in delta not in legacy (extra in delta)
    print("Checking for extra PKs in delta...")
    extra_in_delta = delta_repartitioned.select(primary_key).subtract(
        legacy_repartitioned.select(primary_key)
    ).first()
    has_extra = (extra_in_delta is not None)
    
    pks_match = not (has_missing or has_extra)
    
    if not pks_match:
        print("Step 1 FAILED: PKs do not match")
        results["step1_pk_validation"] = {
            "passed": False
        }
        results["overall_validation_passed"] = False
        results["execution_time_seconds"] = time.time() - start_time
        legacy_df.unpersist()
        delta_df.unpersist()
        return results
    
    results["step1_pk_validation"] = {
        "passed": True
    }
    
    print(f"Step 1 PASSED: PKs match")

    
    # ========================================
    # STEP 2: Content Validation
    # ========================================
    print("Starting Step 2: Content Validation...")
    
    # Get shared columns (excluding primary key and excluded columns)
    legacy_cols = set(legacy_df.columns)
    delta_cols = set(delta_df.columns)
    shared_cols = (legacy_cols & delta_cols) - {primary_key} - set(excluded_columns)
    
    # Inner join since we know PKs match
    print("Joining DataFrames for content validation...")
    joined_df = legacy_repartitioned.alias("legacy").join(
        delta_repartitioned.alias("delta"),
        on=primary_key,
        how="inner"
    )
    
    # Coalesce to reduce partition overhead after join
    # Use 200 partitions to match core count for optimal parallelism
    print("Coalescing joined DataFrame to 200 partitions...")
    joined_df = joined_df.coalesce(200)
    
    # Persist for Step 2 and Step 3
    print("Persisting joined DataFrame...")
    joined_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Use the joined_df directly for content validation
    content_df = joined_df
    
    # Function to compare columns accounting for different types (Step 2)
    def create_comparison_expr(col_name: str, df: DataFrame) -> F.col:
        """Create comparison expression based on column type - completely order-insensitive"""
        col_type = [field.dataType for field in df.schema.fields if field.name == col_name][0]
        
        if isinstance(col_type, ArrayType):
            if col_name == "HOUSING":
                # Get field names from legacy HOUSING schema and sort them
                legacy_housing_fields = sorted([f.name for f in col_type.elementType.fields])
                
                # Build struct with fields in alphabetical order for both legacy and delta
                field_selections = ", ".join([f"x.`{field}` as `{field}`" for field in legacy_housing_fields])
                
                # For legacy: reorder fields alphabetically, convert to JSON, sort array
                legacy_expr = F.expr(f"""
                    array_sort(transform(`legacy`.`{col_name}`, 
                        x -> to_json(struct({field_selections}))
                    ))
                """)
                
                # For delta: filter to only legacy fields, reorder alphabetically, convert to JSON, sort array
                delta_expr = F.expr(f"""
                    array_sort(transform(`delta`.`{col_name}`, 
                        x -> to_json(struct({field_selections}))
                    ))
                """)
                
                return (legacy_expr == delta_expr) | \
                       (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
            
            elif isinstance(col_type.elementType, StructType):
                # For arrays of structs, normalize field order within each struct
                struct_fields = sorted([f.name for f in col_type.elementType.fields])
                field_selections = ", ".join([f"x.`{field}` as `{field}`" for field in struct_fields])
                
                # Reorder fields alphabetically, convert to JSON, sort array
                legacy_expr = F.expr(f"""
                    array_sort(transform(`legacy`.`{col_name}`, 
                        x -> to_json(struct({field_selections}))
                    ))
                """)
                delta_expr = F.expr(f"""
                    array_sort(transform(`delta`.`{col_name}`, 
                        x -> to_json(struct({field_selections}))
                    ))
                """)
                
                return (legacy_expr == delta_expr) | \
                       (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
            else:
                # For arrays of primitives, use simple array_sort
                return (F.array_sort(F.col(f"legacy.{col_name}")) == F.array_sort(F.col(f"delta.{col_name}"))) | \
                       (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
        
        elif isinstance(col_type, StructType):
            # For struct columns, normalize field order before converting to JSON
            struct_fields = sorted([f.name for f in col_type.fields])
            field_selections_legacy = ", ".join([f"`legacy`.`{col_name}`.`{field}` as `{field}`" for field in struct_fields])
            field_selections_delta = ", ".join([f"`delta`.`{col_name}`.`{field}` as `{field}`" for field in struct_fields])
            
            legacy_json = F.expr(f"to_json(struct({field_selections_legacy}))")
            delta_json = F.expr(f"to_json(struct({field_selections_delta}))")
            
            return (legacy_json == delta_json) | \
                   (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
        
        else:
            # For primitive types, direct comparison
            return (F.col(f"legacy.{col_name}") == F.col(f"delta.{col_name}")) | \
                   (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
    
    # Create comparison conditions for all shared columns
    comparison_conditions = []
    for col_name in shared_cols:
        comparison_conditions.append(create_comparison_expr(col_name, legacy_df))
    
    # Combine all conditions
    if comparison_conditions:
        all_match = comparison_conditions[0]
        for condition in comparison_conditions[1:]:
            all_match = all_match & condition
        
        # Check if there are any mismatches by trying to get first mismatch
        validation_df = content_df.withColumn("content_matches", all_match)
        
        # Try to get first mismatched row - if None, no mismatches exist
        mismatched_df = validation_df.filter(F.col("content_matches") == False)
        first_mismatch = mismatched_df.first()
        has_mismatches = (first_mismatch is not None)
    else:
        has_mismatches = False
        first_mismatch = None
    
    content_validation_passed = not has_mismatches
    
    results["step2_content_validation"] = {
        "passed": content_validation_passed
    }
    
    if not content_validation_passed:
        results["overall_validation_passed"] = False
        print(f"\nStep 2 FAILED: Content mismatches detected")
        
        # Use the first_mismatch we already retrieved
        if first_mismatch:
            sample_dict = first_mismatch.asDict()
            print(f"\nSample mismatched row (PK: {sample_dict[primary_key]}):")
            print(f"\nLegacy values:")
            for col in shared_cols:
                print(f"  {col}: {sample_dict.get(f'legacy.{col}')}")
            print(f"\nDelta values:")
            for col in shared_cols:
                print(f"  {col}: {sample_dict.get(f'delta.{col}')}")
        
        # Clean up and return early
        results["execution_time_seconds"] = time.time() - start_time
        joined_df.unpersist()
        legacy_df.unpersist()
        delta_df.unpersist()
        return results
    else:
        print(f"\nStep 2 PASSED: No content mismatches detected")
    
    # Use content_df for Step 3 (only rows with matching PKs)
    step3_df = content_df
    
    # ========================================
    # STEP 3: Assets and INCOME Validation
    # ========================================
    print("Starting Step 3: Assets and INCOME Validation...")
    
    # Check if assets column exists in legacy
    legacy_has_assets = "assets" in legacy_df.columns
    
    if not legacy_has_assets:
        print("Note: legacy.assets column does not exist - skipping Case 2, all rows will be Case 1")
    
    # Check which fields exist in legacy.INCOME schema
    income_schema_fields = []
    for field in legacy_df.schema.fields:
        if field.name == "INCOME" and isinstance(field.dataType, ArrayType):
            income_schema_fields = [f.name for f in field.dataType.elementType.fields]
            break
    
    # Build the qualifying income condition based on which fields exist
    income_conditions = []
    if "ASSETS_AMOUNT" in income_schema_fields:
        income_conditions.append("x.ASSETS_AMOUNT is not null")
    if "UNTAXED_INCOME_AMOUNT" in income_schema_fields:
        income_conditions.append("x.UNTAXED_INCOME_AMOUNT is not null")
    
    # If neither field exists, all Case 1 rows should be Case 1b (delta.assets should be null)
    if not income_conditions:
        print("Note: Neither ASSETS_AMOUNT nor UNTAXED_INCOME_AMOUNT exists in legacy.INCOME schema")
        print("All Case 1 rows will be validated as Case 1b (delta.assets should be null)")
        qualifying_income_condition = "false"  # No qualifying income possible
    else:
        qualifying_income_condition = " or ".join(income_conditions)
    
    # Add columns to categorize each row into cases
    if legacy_has_assets:
        categorized_df = step3_df.withColumn(
            "legacy_assets_valid",
            ~(F.isnull(F.col("legacy.assets")) | (F.size(F.col("legacy.assets")) == 0))
        ).withColumn(
            "has_qualifying_income",
            # Check if any struct in legacy.INCOME has non-null ASSETS_AMOUNT or UNTAXED_INCOME_AMOUNT
            # If fields don't exist, this will always be False
            F.expr(f"exists(`legacy`.`INCOME`, x -> {qualifying_income_condition})")
        )
    else:
        # If legacy.assets doesn't exist, all rows are Case 1 (legacy_assets_valid = False)
        categorized_df = step3_df.withColumn(
            "legacy_assets_valid",
            F.lit(False)
        ).withColumn(
            "has_qualifying_income",
            # Check if any struct in legacy.INCOME has non-null ASSETS_AMOUNT or UNTAXED_INCOME_AMOUNT
            F.expr(f"exists(`legacy`.`INCOME`, x -> {qualifying_income_condition})")
        )
    
    # Case 2: legacy.assets is valid (only if legacy has assets column)
    case2_passed = True
    case2_sample = None
    case2_has_rows = False
    
    if legacy_has_assets:
        case2_df = categorized_df.filter(F.col("legacy_assets_valid") == True)
        # Try to get first row - if None, no rows exist
        case2_has_rows = (case2_df.first() is not None)
        
        # For Case 2, compare legacy.assets with delta.assets (completely order-insensitive)
        if case2_has_rows:
            # Get struct fields from legacy.assets and sort them alphabetically
            assets_schema_fields = None
            for field in legacy_df.schema.fields:
                if field.name == "assets" and isinstance(field.dataType, ArrayType):
                    assets_schema_fields = sorted([f.name for f in field.dataType.elementType.fields])
                    break
            
            if assets_schema_fields:
                # Normalize field order within each struct before comparison
                field_selections = ", ".join([f"x.`{field}` as `{field}`" for field in assets_schema_fields])
                
                case2_comparison = case2_df.withColumn(
                    "assets_match",
                    (F.expr(f"array_sort(transform(`legacy`.`assets`, x -> to_json(struct({field_selections}))))") == 
                     F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({field_selections}))))")) |
                    (F.col("legacy.assets").isNull() & F.col("delta.assets").isNull())
                )
            else:
                # Fallback if we can't get schema
                case2_comparison = case2_df.withColumn(
                    "assets_match",
                    (F.expr("array_sort(transform(`legacy`.`assets`, x -> to_json(x)))") == 
                     F.expr("array_sort(transform(`delta`.`assets`, x -> to_json(x)))")) |
                    (F.col("legacy.assets").isNull() & F.col("delta.assets").isNull())
                )
            
            # Check if any failures exist by trying to get first failure
            case2_failed_df = case2_comparison.filter(F.col("assets_match") == False)
            first_failure = case2_failed_df.first()
            case2_passed = (first_failure is None)
            
            # If failed, use the first_failure we already have
            if not case2_passed:
                case2_sample = first_failure.asDict() if first_failure else None
    
    # Case 1: legacy.assets is null/empty
    case1_df = categorized_df.filter(F.col("legacy_assets_valid") == False)
    
    # Case 1a: has qualifying INCOME structs
    case1a_df = case1_df.filter(F.col("has_qualifying_income") == True)
    case1a_has_rows = (case1a_df.first() is not None)
    
    # Case 1b: no qualifying INCOME structs
    case1b_df = case1_df.filter(F.col("has_qualifying_income") == False)
    case1b_has_rows = (case1b_df.first() is not None)
    
    # For Case 1a: validate that delta.assets contains mapped structs from qualifying legacy.INCOME
    case1a_passed = True
    case1a_sample = None
    if case1a_has_rows:
        # Get the expected field names (delta.assets field names) in alphabetical order
        expected_fields = sorted(income_to_assets_mapping.values())
        
        # Build the struct field mapping expression for the transform
        legacy_to_delta_field_map = {v: k for k, v in income_to_assets_mapping.items()}
        mapped_fields = ", ".join([
            f"x.`{legacy_to_delta_field_map[delta_field]}` as `{delta_field}`"
            for delta_field in expected_fields
        ])
        
        # For delta.assets, also normalize field order
        delta_fields_normalized = ", ".join([f"x.`{field}` as `{field}`" for field in expected_fields])
        
        # Create expected assets by filtering and transforming qualifying INCOME structs
        case1a_validation = case1a_df.withColumn(
            "expected_assets_normalized",
            F.expr(f"""
                array_sort(transform(
                    filter(`legacy`.`INCOME`, x -> {qualifying_income_condition}),
                    x -> to_json(struct({mapped_fields}))
                ))
            """)
        ).withColumn(
            "delta_assets_normalized",
            F.expr(f"""
                array_sort(transform(`delta`.`assets`, 
                    x -> to_json(struct({delta_fields_normalized}))
                ))
            """)
        ).withColumn(
            "assets_match",
            F.col("expected_assets_normalized") == F.col("delta_assets_normalized")
        )
        
        # Check if any failures exist by trying to get first failure
        case1a_failed_df = case1a_validation.filter(F.col("assets_match") == False)
        first_failure = case1a_failed_df.first()
        case1a_passed = (first_failure is None)
        
        # If failed, use the first_failure we already have
        if not case1a_passed:
            case1a_sample = first_failure.asDict() if first_failure else None
    
    # For Case 1b: validate that delta.assets is null or empty
    case1b_passed = True
    case1b_sample = None
    if case1b_has_rows:
        case1b_comparison = case1b_df.withColumn(
            "delta_assets_null_or_empty",
            F.isnull(F.col("delta.assets")) | (F.size(F.col("delta.assets")) == 0)
        )
        case1b_failed_df = case1b_comparison.filter(F.col("delta_assets_null_or_empty") == False)
        first_failure = case1b_failed_df.first()
        case1b_passed = (first_failure is None)
        
        # If failed, use the first_failure we already have
        if not case1b_passed:
            case1b_sample = first_failure.asDict() if first_failure else None
    
    assets_income_validation_passed = (case1a_passed and case1b_passed and case2_passed)
    
    results["step3_assets_income_validation"] = {
        "passed": assets_income_validation_passed,
        "case1a_passed": case1a_passed,
        "case1b_passed": case1b_passed,
        "case2_passed": case2_passed
    }
    
    if not assets_income_validation_passed:
        results["overall_validation_passed"] = False
        print(f"\nStep 3 FAILED: Assets/INCOME validation issues detected")
        
        if not case1a_passed and case1a_sample:
            print(f"\n  Case 1a FAILED (PK: {case1a_sample[primary_key]}):")
            print(f"    legacy.INCOME: {case1a_sample.get('INCOME')}")
            print(f"    Expected delta.assets: {case1a_sample.get('expected_assets_normalized')}")
            print(f"    Actual delta.assets: {case1a_sample.get('delta_assets_normalized')}")
        
        if not case1b_passed and case1b_sample:
            print(f"\n  Case 1b FAILED (PK: {case1b_sample[primary_key]}):")
            print(f"    legacy.INCOME: {case1b_sample.get('INCOME')}")
            print(f"    legacy.assets: {case1b_sample.get('assets')}")
            print(f"    delta.assets should be null/empty but is: {case1b_sample.get('assets')}")
        
        if not case2_passed and case2_sample:
            print(f"\n  Case 2 FAILED (PK: {case2_sample[primary_key]}):")
            print(f"    legacy.assets: {case2_sample.get('assets')}")
            print(f"    delta.assets: {case2_sample.get('assets')}")
    else:
        print(f"\nStep 3 PASSED: All assets and INCOME validations successful")
    
    # Unpersist cached DataFrames
    joined_df.unpersist()
    legacy_df.unpersist()
    delta_df.unpersist()
    
    results["execution_time_seconds"] = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Validation Complete in {results['execution_time_seconds']:.2f} seconds")
    print(f"Overall Result: {'PASSED' if results['overall_validation_passed'] else 'FAILED'}")
    print(f"{'='*50}")
    
    return results


# Example usage:
if __name__ == "__main__":
    # This is just a template - you would load your actual DataFrames
    # legacy_df = spark.read.parquet("path/to/legacy")
    # delta_df = spark.read.format("delta").load("path/to/delta")
    
    # Run validation
    # results = compare_legacy_vs_delta(legacy_df, delta_df, "id")
    # print(results)
    
    # If validation fails, check the sample mismatch
    # if not results["step2_content_validation"]["content_validation_passed"]:
    #     sample = results["step2_content_validation"]["sample_mismatch"]
    #     print(f"\nSample mismatch for PK: {sample['id']}")
    #     print(f"\nLegacy row: {sample['legacy_row']}")
    #     print(f"\nDelta row: {sample['delta_row']}")
    pass
