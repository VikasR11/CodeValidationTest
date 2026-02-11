from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType
from typing import Dict, List
import time


def validate_dataframes(legacy_df: DataFrame, delta_df: DataFrame, primary_key: str) -> Dict:
    """
    Validate that delta DataFrame matches legacy DataFrame (source of truth) for 230M rows.
    
    Args:
        legacy_df: Source of truth DataFrame in Parquet format
        delta_df: Modern DataFrame in Delta Lake format
        primary_key: Single column name to use as primary key
    
    Returns:
        Dictionary with validation results for all steps
    """
    
    start_time = time.time()
    
    # Define excluded columns for step 2 (handled in step 3)
    excluded_columns = ['assets', 'INCOME']
    
    # Define income to assets field mapping for Case 1a
    # TODO: Fill in complete mapping
    income_to_assets_mapping = {
        'INCOME_SOURCE': 'assets_source',
        'INCOME_CHANNEL': 'assets_channel',
        # Add more mappings here
    }
    
    # Cache DataFrames for multiple operations
    legacy_df.cache()
    delta_df.cache()
    
    results = {
        "step1_pk_validation": {},
        "step2_content_validation": {},
        "step3_assets_income_validation": {},
        "overall_validation_passed": True,
        "execution_time_seconds": 0
    }
    
    # ========================================
    # STEPS 1 & 2 COMBINED: Single Join Operation
    # ========================================
    print("Starting Steps 1 & 2: PK Validation and Content Validation...")
    
    # Get shared columns (excluding primary key and excluded columns)
    legacy_cols = set(legacy_df.columns)
    delta_cols = set(delta_df.columns)
    shared_cols = (legacy_cols & delta_cols) - {primary_key} - set(excluded_columns)
    
    # Perform full outer join to check PKs and content in one operation
    joined_df = legacy_df.alias("legacy").join(
        delta_df.alias("delta"),
        on=primary_key,
        how="full_outer"
    )
    
    # Add indicators for PK validation (Step 1)
    joined_df = joined_df.withColumn(
        "missing_in_delta", 
        F.col(f"delta.{primary_key}").isNull()
    ).withColumn(
        "extra_in_delta",
        F.col(f"legacy.{primary_key}").isNull()
    )
    
    # Count PK validation metrics (Step 1)
    pk_stats = joined_df.agg(
        F.sum(F.when(F.col("missing_in_delta"), 1).otherwise(0)).alias("missing_in_delta"),
        F.sum(F.when(F.col("extra_in_delta"), 1).otherwise(0)).alias("extra_in_delta"),
        F.count(F.when(~F.col("missing_in_delta") & ~F.col("extra_in_delta"), 1)).alias("matching_count")
    ).collect()[0]
    
    missing_in_delta = pk_stats["missing_in_delta"]
    extra_in_delta = pk_stats["extra_in_delta"]
    matching_count = pk_stats["matching_count"]
    
    pk_validation_passed = (missing_in_delta == 0 and extra_in_delta == 0)
    
    results["step1_pk_validation"] = {
        "legacy_count": matching_count if pk_validation_passed else matching_count + missing_in_delta,
        "delta_count": matching_count if pk_validation_passed else matching_count + extra_in_delta,
        "missing_in_delta": missing_in_delta,
        "extra_in_delta": extra_in_delta,
        "pk_validation_passed": pk_validation_passed
    }
    
    if not pk_validation_passed:
        results["overall_validation_passed"] = False
        print(f"Step 1 FAILED: {missing_in_delta} missing in delta, {extra_in_delta} extra in delta")
        results["execution_time_seconds"] = time.time() - start_time
        legacy_df.unpersist()
        delta_df.unpersist()
        return results
    
    print(f"Step 1 PASSED: All {matching_count} PKs match")
    
    # Filter to only matching PKs for content validation (Step 2)
    content_df = joined_df.filter(~F.col("missing_in_delta") & ~F.col("extra_in_delta"))
    
    # Function to compare columns accounting for different types (Step 2)
    def create_comparison_expr(col_name: str, df: DataFrame) -> F.col:
        """Create comparison expression based on column type"""
        col_type = [field.dataType for field in df.schema.fields if field.name == col_name][0]
        
        if isinstance(col_type, ArrayType):
            # For arrays (including array of structs), convert to JSON, sort, and compare
            # Special handling for HOUSING array to ignore HOUSING_TYPE field in delta
            if col_name == "HOUSING":
                # Get field names from legacy HOUSING schema (excluding HOUSING_TYPE)
                legacy_housing_fields = [f.name for f in col_type.elementType.fields]
                
                # Create expression to transform delta.HOUSING, selecting only fields that exist in legacy
                field_selections = ", ".join([f"x.{field} as {field}" for field in legacy_housing_fields])
                delta_filtered = F.expr(f"transform(`delta.HOUSING`, x -> struct({field_selections}))")
                
                # Compare sorted JSON (order doesn't matter)
                legacy_json = F.to_json(F.array_sort(F.col(f"legacy.{col_name}")))
                delta_json = F.to_json(F.array_sort(delta_filtered))
                return (legacy_json == delta_json) | (F.col(f"legacy.{col_name}").isNull() & delta_filtered.isNull())
            else:
                # Convert arrays to sorted JSON strings for comparison (order doesn't matter)
                legacy_json = F.to_json(F.array_sort(F.col(f"legacy.{col_name}")))
                delta_json = F.to_json(F.array_sort(F.col(f"delta.{col_name}")))
                return (legacy_json == delta_json) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
        
        elif isinstance(col_type, StructType):
            # For struct columns, convert to JSON and compare
            legacy_json = F.to_json(F.col(f"legacy.{col_name}"))
            delta_json = F.to_json(F.col(f"delta.{col_name}"))
            return (legacy_json == delta_json) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
        
        else:
            # For primitive types, direct comparison
            return (F.col(f"legacy.{col_name}") == F.col(f"delta.{col_name}")) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
    
    # Create comparison conditions for all shared columns
    comparison_conditions = []
    for col_name in shared_cols:
        comparison_conditions.append(create_comparison_expr(col_name, legacy_df))
    
    # Combine all conditions
    if comparison_conditions:
        all_match = comparison_conditions[0]
        for condition in comparison_conditions[1:]:
            all_match = all_match & condition
        
        # Count matching and mismatched rows
        validation_df = content_df.withColumn("content_matches", all_match)
        matching_rows = validation_df.filter(F.col("content_matches") == True).count()
        mismatched_rows = validation_df.filter(F.col("content_matches") == False).count()
        
        # If there are mismatches, get one sample row
        sample_mismatch = None
        if mismatched_rows > 0:
            sample_row = validation_df.filter(F.col("content_matches") == False).first()
            if sample_row:
                sample_dict = sample_row.asDict()
                sample_mismatch = {
                    primary_key: sample_dict[primary_key],
                    "legacy_row": {k.replace("legacy.", ""): v for k, v in sample_dict.items() if k.startswith("legacy.")},
                    "delta_row": {k.replace("delta.", ""): v for k, v in sample_dict.items() if k.startswith("delta.")}
                }
    else:
        matching_rows = matching_count
        mismatched_rows = 0
        sample_mismatch = None
    
    content_validation_passed = (mismatched_rows == 0)
    
    results["step2_content_validation"] = {
        "rows_checked": matching_count,
        "matching_rows": matching_rows,
        "mismatched_rows": mismatched_rows,
        "content_validation_passed": content_validation_passed,
        "sample_mismatch": sample_mismatch
    }
    
    if not content_validation_passed:
        results["overall_validation_passed"] = False
        print(f"Step 2 FAILED: {mismatched_rows} rows have content mismatches")
        if sample_mismatch:
            print(f"Sample mismatched row PK: {sample_mismatch[primary_key]}")
    else:
        print(f"Step 2 PASSED: All {matching_rows} rows match")
    
    # Use content_df for Step 3 (only rows with matching PKs)
    step3_df = content_df
    
    # ========================================
    # STEP 3: Assets and INCOME Validation
    # ========================================
    print("Starting Step 3: Assets and INCOME Validation...")
    
    # Add columns to categorize each row into cases
    categorized_df = step3_df.withColumn(
        "legacy_assets_valid",
        ~(F.isnull(F.col("legacy.assets")) | (F.size(F.col("legacy.assets")) == 0))
    ).withColumn(
        "has_qualifying_income",
        # Check if any struct in legacy.INCOME has non-null ASSETS_AMOUNT or UNTAXED_INCOME_AMOUNT
        F.expr("exists(`legacy.INCOME`, x -> x.ASSETS_AMOUNT is not null or x.UNTAXED_INCOME_AMOUNT is not null)")
    )
    
    # Case 2: legacy.assets is valid
    case2_df = categorized_df.filter(F.col("legacy_assets_valid") == True)
    case2_total = case2_df.count()
    
    # For Case 2, compare legacy.assets with delta.assets (order doesn't matter)
    # Convert to sorted JSON for comparison
    if case2_total > 0:
        case2_comparison = case2_df.withColumn(
            "assets_match",
            (F.to_json(F.array_sort(F.col("legacy.assets"))) == F.to_json(F.array_sort(F.col("delta.assets")))) |
            (F.col("legacy.assets").isNull() & F.col("delta.assets").isNull())
        )
        case2_passed = case2_comparison.filter(F.col("assets_match") == True).count()
        case2_failed = case2_total - case2_passed
    else:
        case2_passed = 0
        case2_failed = 0
    
    # Case 1: legacy.assets is null/empty
    case1_df = categorized_df.filter(F.col("legacy_assets_valid") == False)
    
    # Case 1a: has qualifying INCOME structs
    case1a_df = case1_df.filter(F.col("has_qualifying_income") == True)
    case1a_total = case1a_df.count()
    
    # Case 1b: no qualifying INCOME structs
    case1b_df = case1_df.filter(F.col("has_qualifying_income") == False)
    case1b_total = case1b_df.count()
    
    # For Case 1a: validate that delta.assets contains mapped structs from qualifying legacy.INCOME
    # Filter qualifying INCOME structs and map to expected assets format
    if case1a_total > 0:
        # Build the struct field mapping expression for the transform
        mapped_fields = ", ".join([
            f"x.{legacy_field} as {delta_field}" 
            for legacy_field, delta_field in income_to_assets_mapping.items()
        ])
        
        # Create expected assets by filtering and transforming qualifying INCOME structs
        case1a_validation = case1a_df.withColumn(
            "expected_assets",
            F.expr(f"""
                transform(
                    filter(`legacy.INCOME`, x -> x.ASSETS_AMOUNT is not null or x.UNTAXED_INCOME_AMOUNT is not null),
                    x -> struct({mapped_fields})
                )
            """)
        ).withColumn(
            "assets_match",
            # Compare sorted JSON arrays (order doesn't matter)
            F.to_json(F.array_sort(F.col("expected_assets"))) == F.to_json(F.array_sort(F.col("delta.assets")))
        )
        
        case1a_passed = case1a_validation.filter(F.col("assets_match") == True).count()
        case1a_failed = case1a_total - case1a_passed
    else:
        case1a_passed = 0
        case1a_failed = 0
    
    # For Case 1b: validate that delta.assets is null
    if case1b_total > 0:
        case1b_comparison = case1b_df.withColumn(
            "delta_assets_null",
            F.isnull(F.col("delta.assets"))
        )
        case1b_passed = case1b_comparison.filter(F.col("delta_assets_null") == True).count()
        case1b_failed = case1b_total - case1b_passed
    else:
        case1b_passed = 0
        case1b_failed = 0
    
    assets_income_validation_passed = (case1a_failed == 0 and case1b_failed == 0 and case2_failed == 0)
    
    results["step3_assets_income_validation"] = {
        "case1a": {"total": case1a_total, "passed": case1a_passed, "failed": case1a_failed},
        "case1b": {"total": case1b_total, "passed": case1b_passed, "failed": case1b_failed},
        "case2": {"total": case2_total, "passed": case2_passed, "failed": case2_failed},
        "validation_passed": assets_income_validation_passed
    }
    
    if not assets_income_validation_passed:
        results["overall_validation_passed"] = False
        print(f"Step 3 FAILED: Assets/INCOME validation issues detected")
    else:
        print(f"Step 3 PASSED: All assets and INCOME validations successful")
    
    # Unpersist cached DataFrames
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
    # results = validate_dataframes(legacy_df, delta_df, "id")
    # print(results)
    
    # If validation fails, check the sample mismatch
    # if not results["step2_content_validation"]["content_validation_passed"]:
    #     sample = results["step2_content_validation"]["sample_mismatch"]
    #     print(f"\nSample mismatch for PK: {sample['id']}")
    #     print(f"\nLegacy row: {sample['legacy_row']}")
    #     print(f"\nDelta row: {sample['delta_row']}")
    pass
