# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Property Price GBT Prediction") \
    .getOrCreate()

print("Starting property price analysis with GBT...")

# Load data from HDFS
try:
    df = spark.read.csv("hdfs://namenode:8020/ProjetBigData/data.csv", header=True, inferSchema=True)
    print("Successfully loaded data from HDFS")
except Exception as e:
    print("Failed to load data from HDFS: " + str(e))
    # Try alternate locations
    try:
        # Try local filesystem
        df = spark.read.csv("/tmp/data.csv", header=True, inferSchema=True)
        print("Successfully loaded data from local filesystem")
    except Exception as e2:
        print("Failed to load from local filesystem: " + str(e2))
        # Try other possible filenames (case sensitivity matters)
        try:
            df = spark.read.csv("/tmp/data.Scv", header=True, inferSchema=True)
            print("Successfully loaded data.Scv from local filesystem")
        except Exception as e3:
            print("Could not load any data: " + str(e3))
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
    # Try to find a column that might be the price
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

# GBT Regressor with optimized parameters
gbt_regressor = GBTRegressor(
    featuresCol="features", 
    labelCol=log_target_col,
    maxIter=100,         # Reduced from 200 for faster execution
    maxDepth=5,          # Reduced from 8 for faster execution
    stepSize=0.05,       
    subsamplingRate=0.8,  
    featureSubsetStrategy="sqrt",
    maxBins=32,          # Reduced from 64 for faster execution
    minInstancesPerNode=1,
    seed=42               
)
print("Configured GBT Regressor with optimized parameters")

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

# Try to extract feature importance
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

# Stop Spark session
spark.stop()
