from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType
from pyspark import StorageLevel
from typing import Dict
import time


def compare_legacy_vs_delta(legacy_df: DataFrame, delta_df: DataFrame, primary_key: str, materialize_inputs: bool = True) -> Dict:
    """Validate delta DataFrame matches legacy DataFrame (source of truth)."""
    start_time = time.time()
    
    if materialize_inputs:
        legacy_df.persist(StorageLevel.MEMORY_AND_DISK)
        delta_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    excluded_columns = ['assets', 'INCOME']
    income_to_assets_mapping = {
        'INCM_SOURCE': 'assets_source',
        'INCOME_CHANNEL_NAME': 'assets_channel_name',
        'STRATEGY_NAME': 'assets_strategy_name',
        'INCOME_LAST_MODIFIED_SOURCE_SYSTEM': 'assets_last_modified_source_system',
        'INCOME_REPORTED_TS': 'assets_reported_utc_timestamp',
        'ASSETS_AMOUNT': 'assets_amount',
        'UNTAXED_INCOME_AMOUNT': 'untaxed_income_amount',
    }
    
    results = {"step1_pk_validation": {}, "step2_content_validation": {}, "step3_assets_income_validation": {}, "overall_validation_passed": True, "execution_time_seconds": 0}
    
    # STEP 1: PK Validation
    print("Step 1: PK Validation...")
    missing = legacy_df.select(primary_key).subtract(delta_df.select(primary_key)).first()
    extra = delta_df.select(primary_key).subtract(legacy_df.select(primary_key)).first()
    pks_match = not (missing is not None or extra is not None)
    
    results["step1_pk_validation"] = {"passed": pks_match}
    if not pks_match:
        results["overall_validation_passed"] = False
        results["execution_time_seconds"] = time.time() - start_time
        legacy_df.unpersist()
        delta_df.unpersist()
        print("Step 1 FAILED")
        return results
    print("Step 1 PASSED")
    
    # STEP 2: Content Validation
    print("Step 2: Content Validation...")
    shared_cols = (set(legacy_df.columns) & set(delta_df.columns)) - {primary_key} - set(excluded_columns)
    joined_df = legacy_df.alias("legacy").join(delta_df.alias("delta"), on=primary_key, how="inner").coalesce(200)
    joined_df.persist(StorageLevel.MEMORY_AND_DISK)
    
    def create_comparison_expr(col_name: str, df: DataFrame) -> F.col:
        col_type = [f.dataType for f in df.schema.fields if f.name == col_name][0]
        
        if isinstance(col_type, ArrayType):
            if col_name == "HOUSING":
                fields = sorted([f.name for f in col_type.elementType.fields])
                fs = ", ".join([f"x.`{f}` as `{f}`" for f in fields])
                return (F.expr(f"array_sort(transform(`legacy`.`{col_name}`, x -> to_json(struct({fs}))))") == F.expr(f"array_sort(transform(`delta`.`{col_name}`, x -> to_json(struct({fs}))))")) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
            elif isinstance(col_type.elementType, StructType):
                fields = sorted([f.name for f in col_type.elementType.fields])
                fs = ", ".join([f"x.`{f}` as `{f}`" for f in fields])
                return (F.expr(f"array_sort(transform(`legacy`.`{col_name}`, x -> to_json(struct({fs}))))") == F.expr(f"array_sort(transform(`delta`.`{col_name}`, x -> to_json(struct({fs}))))")) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
            else:
                return (F.array_sort(F.col(f"legacy.{col_name}")) == F.array_sort(F.col(f"delta.{col_name}"))) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
        elif isinstance(col_type, StructType):
            fields = sorted([f.name for f in col_type.fields])
            fs_leg = ", ".join([f"`legacy`.`{col_name}`.`{f}` as `{f}`" for f in fields])
            fs_del = ", ".join([f"`delta`.`{col_name}`.`{f}` as `{f}`" for f in fields])
            return (F.expr(f"to_json(struct({fs_leg}))") == F.expr(f"to_json(struct({fs_del}))")) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
        else:
            return (F.col(f"legacy.{col_name}") == F.col(f"delta.{col_name}")) | (F.col(f"legacy.{col_name}").isNull() & F.col(f"delta.{col_name}").isNull())
    
    comparison_conditions = [create_comparison_expr(c, legacy_df) for c in shared_cols]
    
    if comparison_conditions:
        all_match = comparison_conditions[0]
        for cond in comparison_conditions[1:]:
            all_match = all_match & cond
        mismatched = joined_df.withColumn("content_matches", all_match).filter(F.col("content_matches") == False).first()
        has_mismatches = (mismatched is not None)
    else:
        has_mismatches = False
        mismatched = None
    
    results["step2_content_validation"] = {"passed": not has_mismatches}
    if has_mismatches:
        results["overall_validation_passed"] = False
        print("Step 2 FAILED")
        if mismatched:
            sd = mismatched.asDict()
            print(f"Sample (PK: {sd[primary_key]})")
            print("Legacy:", {c: sd.get(f'legacy.{c}') for c in shared_cols})
            print("Delta:", {c: sd.get(f'delta.{c}') for c in shared_cols})
        results["execution_time_seconds"] = time.time() - start_time
        joined_df.unpersist()
        legacy_df.unpersist()
        delta_df.unpersist()
        return results
    print("Step 2 PASSED")
    
    # STEP 3: Assets/INCOME Validation
    print("Step 3: Assets/INCOME Validation...")
    legacy_has_assets = "assets" in legacy_df.columns
    
    income_fields = []
    for f in legacy_df.schema.fields:
        if f.name == "INCOME" and isinstance(f.dataType, ArrayType):
            income_fields = [sf.name for sf in f.dataType.elementType.fields]
            break
    
    income_conds = []
    if "ASSETS_AMOUNT" in income_fields:
        income_conds.append("x.ASSETS_AMOUNT is not null")
    if "UNTAXED_INCOME_AMOUNT" in income_fields:
        income_conds.append("x.UNTAXED_INCOME_AMOUNT is not null")
    qualifying = " or ".join(income_conds) if income_conds else "false"
    
    if legacy_has_assets:
        cat = joined_df.withColumn("leg_valid", ~(F.isnull(F.col("legacy.assets")) | (F.size(F.col("legacy.assets")) == 0))).withColumn("has_qual", F.expr(f"exists(`legacy`.`INCOME`, x -> {qualifying})"))
    else:
        cat = joined_df.withColumn("leg_valid", F.lit(False)).withColumn("has_qual", F.expr(f"exists(`legacy`.`INCOME`, x -> {qualifying})"))
    
    # Case 2
    case2_passed = True
    if legacy_has_assets:
        c2 = cat.filter(F.col("leg_valid") == True)
        if c2.first():
            asset_fields = None
            for f in legacy_df.schema.fields:
                if f.name == "assets" and isinstance(f.dataType, ArrayType):
                    asset_fields = sorted([sf.name for sf in f.dataType.elementType.fields])
                    break
            if asset_fields:
                fs = ", ".join([f"x.`{f}` as `{f}`" for f in asset_fields])
                c2_comp = c2.withColumn("match", (F.expr(f"array_sort(transform(`legacy`.`assets`, x -> to_json(struct({fs}))))") == F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({fs}))))")) | (F.col("legacy.assets").isNull() & F.col("delta.assets").isNull()))
                case2_passed = (c2_comp.filter(F.col("match") == False).first() is None)
    
    # Case 1a
    c1a = cat.filter((F.col("leg_valid") == False) & (F.col("has_qual") == True))
    case1a_passed = True
    if c1a.first():
        exp_fields = sorted(income_to_assets_mapping.values())
        l2d = {v: k for k, v in income_to_assets_mapping.items()}
        mapped = ", ".join([f"x.`{l2d[d]}` as `{d}`" for d in exp_fields])
        delta_norm = ", ".join([f"x.`{f}` as `{f}`" for f in exp_fields])
        c1a_val = c1a.withColumn("exp", F.expr(f"array_sort(transform(filter(`legacy`.`INCOME`, x -> {qualifying}), x -> to_json(struct({mapped}))))")).withColumn("del_norm", F.expr(f"array_sort(transform(`delta`.`assets`, x -> to_json(struct({delta_norm}))))")).withColumn("match", F.col("exp") == F.col("del_norm"))
        case1a_passed = (c1a_val.filter(F.col("match") == False).first() is None)
    
    # Case 1b
    c1b = cat.filter((F.col("leg_valid") == False) & (F.col("has_qual") == False))
    case1b_passed = True
    if c1b.first():
        c1b_comp = c1b.withColumn("null_empty", F.isnull(F.col("delta.assets")) | (F.size(F.col("delta.assets")) == 0))
        case1b_passed = (c1b_comp.filter(F.col("null_empty") == False).first() is None)
    
    step3_passed = (case1a_passed and case1b_passed and case2_passed)
    results["step3_assets_income_validation"] = {"passed": step3_passed, "case1a_passed": case1a_passed, "case1b_passed": case1b_passed, "case2_passed": case2_passed}
    
    if not step3_passed:
        results["overall_validation_passed"] = False
        print("Step 3 FAILED")
    else:
        print("Step 3 PASSED")
    
    joined_df.unpersist()
    legacy_df.unpersist()
    delta_df.unpersist()
    results["execution_time_seconds"] = time.time() - start_time
    print(f"Validation {'PASSED' if results['overall_validation_passed'] else 'FAILED'} ({results['execution_time_seconds']:.2f}s)")
    return results


def compare_legacy_vs_delta_chunked(legacy_df: DataFrame, delta_df: DataFrame, primary_key: str, num_chunks: int = 4, materialize_inputs: bool = True) -> Dict:
    """Validate in chunks - more reliable for large DataFrames."""
    start_time = time.time()
    print(f"\n{'='*60}\nCHUNKED: {num_chunks} chunks\n{'='*60}")
    
    total = legacy_df.count()
    print(f"Total: {total:,}\n")
    
    results = {"total_rows": total, "num_chunks": num_chunks, "chunks_validated": 0, "chunk_results": [], "overall_validation_passed": True, "execution_time_seconds": 0}
    
    for i in range(num_chunks):
        print(f"\n{'='*60}\nCHUNK {i+1}/{num_chunks}\n{'='*60}")
        
        chunk_leg = legacy_df.filter(F.expr(f"abs(hash({primary_key})) % {num_chunks} = {i}"))
        chunk_cnt = chunk_leg.count()
        print(f"Size: {chunk_cnt:,}")
        
        chunk_del = delta_df.join(chunk_leg.select(primary_key), on=primary_key, how="inner")
        chunk_res = compare_legacy_vs_delta(chunk_leg, chunk_del, primary_key, materialize_inputs)
        
        results["chunk_results"].append({"chunk_num": i+1, "chunk_size": chunk_cnt, "passed": chunk_res["overall_validation_passed"], "execution_time": chunk_res["execution_time_seconds"]})
        results["chunks_validated"] += 1
        
        if not chunk_res["overall_validation_passed"]:
            print(f"\n❌ CHUNK {i+1} FAILED")
            results["overall_validation_passed"] = False
            results["execution_time_seconds"] = time.time() - start_time
            return results
        print(f"✓ Chunk {i+1} PASSED")
    
    results["execution_time_seconds"] = time.time() - start_time
    print(f"\n{'='*60}\n✓ ALL PASSED ({results['execution_time_seconds']:.2f}s)\n{'='*60}")
    return results
