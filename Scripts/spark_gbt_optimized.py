from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, stddev, mean
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Enhanced Property Price GBT Prediction") \
    .getOrCreate()

print("Starting enhanced property price analysis with GBT...")

# Load data from HDFS
try:
    df = spark.read.csv("hdfs://namenode:8020/ProjetBigData/data.csv", header=True, inferSchema=True)
    print("Successfully loaded data from HDFS")
except Exception as e:
    print("Failed to load data from HDFS: " + str(e))
    try:
        df = spark.read.csv("/tmp/data.csv", header=True, inferSchema=True)
        print("Successfully loaded data from local filesystem")
    except Exception as e2:
        print("Could not load any data: " + str(e2))
        import sys
        sys.exit(1)

# Print dataset info
print("Dataset shape: (" + str(df.count()) + ", " + str(len(df.columns)) + ")")

# Identify string columns that need to be indexed
string_cols = [field.name for field in df.schema.fields 
              if field.dataType.typeName() == "string"]
print("Found " + str(len(string_cols)) + " string columns to index")

# Define target column
target_col = "price"

# Check if target column exists
if target_col not in df.columns:
    print("WARNING: Target column 'price' not found in dataset.")
    numeric_cols = [field.name for field in df.schema.fields 
                   if field.dataType.typeName() in ["integer", "double", "long"]]
    if len(numeric_cols) > 0:
        target_col = numeric_cols[0]
        print("Using " + target_col + " as the target column instead")
    else:
        print("No suitable numeric column found for prediction.")
        import sys
        sys.exit(1)

# Create log transformation of target
print("Creating log transformation of target column: " + target_col)
df = df.withColumn("log_price", F.log1p(df[target_col]))
log_target_col = "log_price"

# Define feature columns (excluding target and string columns)
feature_cols = [col for col in df.columns 
               if col != target_col and col != log_target_col and col not in string_cols]
print("Using " + str(len(feature_cols)) + " numeric features")

# Handle missing values
print("Handling missing values...")
for col_name in feature_cols:
    df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), F.lit(0)).otherwise(F.col(col_name)))

# IMPROVEMENT 3: Filter outliers based on statistics
print("Filtering outliers...")
# Calculate statistics for the target variable
price_stats = df.select(
    mean(log_target_col).alias("mean"),
    stddev(log_target_col).alias("stddev")
).collect()[0]

# Define bounds for outlier detection (3 standard deviations)
mean_val = price_stats["mean"]
stddev_val = price_stats["stddev"]
lower_bound = mean_val - 3 * stddev_val
upper_bound = mean_val + 3 * stddev_val

# Filter out outliers
df_filtered = df.filter(
    (F.col(log_target_col) >= lower_bound) & 
    (F.col(log_target_col) <= upper_bound)
)

print(f"Filtered out {df.count() - df_filtered.count()} outliers")
df = df_filtered

# IMPROVEMENT 1: Advanced feature engineering
print("Performing advanced feature engineering...")

# List of important features for creating interactions and transformations
important_features = ["living_area", "bedrooms", "bathrooms", "total_rooms", "land_area"]
important_features = [f for f in important_features if f in feature_cols]  # Keep only existing columns

# Create squared terms for important features
for feat in important_features:
    squared_col = feat + "_squared"
    df = df.withColumn(squared_col, F.col(feat) * F.col(feat))
    feature_cols.append(squared_col)
    print(f"Created squared feature: {squared_col}")

# Create ratio features (for top feature pairs)
if "living_area" in feature_cols and "land_area" in feature_cols:
    df = df.withColumn("living_land_ratio", 
                     F.when(F.col("land_area") > 0, F.col("living_area") / F.col("land_area")).otherwise(0))
    feature_cols.append("living_land_ratio")
    print("Created ratio feature: living_land_ratio")

if "bedrooms" in feature_cols and "bathrooms" in feature_cols:
    df = df.withColumn("bed_bath_ratio", 
                     F.when(F.col("bathrooms") > 0, F.col("bedrooms") / F.col("bathrooms")).otherwise(0))
    feature_cols.append("bed_bath_ratio")
    print("Created ratio feature: bed_bath_ratio")

if "living_area" in feature_cols and "bedrooms" in feature_cols:
    df = df.withColumn("area_per_bedroom", 
                     F.when(F.col("bedrooms") > 0, F.col("living_area") / F.col("bedrooms")).otherwise(0))
    feature_cols.append("area_per_bedroom")
    print("Created ratio feature: area_per_bedroom")

# Create interaction terms (limit to 3 important combinations to avoid explosion of features)
key_interactions = [
    ("living_area", "bathrooms"),
    ("bedrooms", "bathrooms"),
    ("living_area", "land_area")
]

for feat1, feat2 in key_interactions:
    if feat1 in feature_cols and feat2 in feature_cols:
        interaction_col = f"{feat1}_x_{feat2}"
        df = df.withColumn(interaction_col, F.col(feat1) * F.col(feat2))
        feature_cols.append(interaction_col)
        print(f"Created interaction feature: {interaction_col}")

# Create StringIndexers for categorical columns
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep") 
           for col_name in string_cols]
print("Created indexers for " + str(len(indexers)) + " categorical columns")

# Create vector assembler
assembler = VectorAssembler(
    inputCols=feature_cols + [c + "_indexed" for c in string_cols], 
    outputCol="features", 
    handleInvalid="skip"
)

# IMPROVEMENT 2: Optimized GBT Regressor
gbt_regressor = GBTRegressor(
    featuresCol="features", 
    labelCol=log_target_col,
    maxIter=150,         # Increased from 100 to 150
    maxDepth=7,          # Increased from 5 to 7
    stepSize=0.03,       # Reduced from 0.05 to 0.03 for more precision
    subsamplingRate=0.7,  # Adjusted from 0.8 to 0.7
    featureSubsetStrategy="all",  # Use all features instead of "sqrt"
    maxBins=40,          # Increased from 32 to 40
    minInstancesPerNode=1,
    seed=42               
)
print("Configured GBT Regressor with enhanced parameters")

# Create pipeline
pipeline = Pipeline(stages=indexers + [assembler, gbt_regressor])

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print("Split data: Train set size: " + str(train_df.count()) + 
      ", Test set size: " + str(test_df.count()))

# Train model
print("Training GBT model...")
model = pipeline.fit(train_df)
print("Model training complete")

# Make predictions
print("Making predictions on test set...")
predictions = model.transform(test_df)

# Convert log predictions back to original scale
predictions = predictions.withColumn("exp_prediction", F.expm1(predictions["prediction"]))

# Evaluate model on log scale
print("Evaluating model...")
evaluator = RegressionEvaluator(labelCol=log_target_col, predictionCol="prediction", metricName="r2")
r2_log = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(labelCol=log_target_col, predictionCol="prediction", metricName="rmse")
rmse_log = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(labelCol=log_target_col, predictionCol="prediction", metricName="mae")
mae_log = evaluator.evaluate(predictions)

# Evaluate model on original scale
evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="exp_prediction", metricName="r2")
r2_original = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="exp_prediction", metricName="rmse")
rmse_original = evaluator.evaluate(predictions)

# Print metrics
print("\nLog Scale Metrics:")
print("R2 score: {:.4f}".format(r2_log))
print("RMSE: {:.4f}".format(rmse_log))
print("MAE: {:.4f}".format(mae_log))

print("\nOriginal Scale Metrics:")
print("R2 score: {:.4f}".format(r2_original))
print("RMSE: {:.4f}".format(rmse_original))

# Extract feature importance
try:
    print("\n===== FEATURE IMPORTANCE =====")
    gbt_model = model.stages[-1]
    importances = gbt_model.featureImportances
    
    # Get feature names
    feature_names = feature_cols + [c + "_indexed" for c in string_cols]
    
    # Create feature importance pairs
    feature_importance = [(feature, float(importance)) 
                         for feature, importance in zip(feature_names, importances)]
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 features:")
    for i, (feature, importance) in enumerate(feature_importance[:15]):
        print("{}. {}: {:.4f}".format(i+1, feature, importance))
except Exception as e:
    print("Could not extract feature importance: " + str(e))

# Calculate adjusted R-squared
feature_count = len(feature_cols + string_cols)
n = predictions.count()
adjusted_r2 = 1 - ((1 - r2_log) * (n - 1) / (n - feature_count - 1))
print("\nAdvanced Metrics:")
print("Adjusted R2 (log scale): {:.4f}".format(adjusted_r2))
print("Feature count: {}".format(feature_count))

# Stop Spark session
spark.stop()
