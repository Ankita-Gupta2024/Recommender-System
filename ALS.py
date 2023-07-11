import csv
import getpass
import itertools
import sys
import time
import os
from hdfs import InsecureClient
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import collect_list, explode, count, col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType



def evaluation(popular_tracks_df, model, validation_data, top_n=100):
    
    #Group the trained_n_validated_data by user_id and collect their recording_msid into a list
    popular_tracks_df = popular_tracks_df.groupBy('user_id').agg(collect_list('track_id_index').alias('actual'))

    #top 100 recommendations for each user as per existing ALS
    users = validation_data.select('user_id').distinct()
    recommended = model.recommendForUserSubset(users, top_n)
    predicted_using_model_data = (recommended.select("user_id", explode("recommendations").alias("recommendation")).select("user_id", "recommendation.*"))

    # Group the out data by user_id and collect their recording_msid into a list as 'predicted' column 
    user_relevant_tracks = predicted_using_model_data.groupBy('user_id').agg(collect_list('track_id_index').alias('predicted'))

    #create a new dataframe with user_id, actual and predicted columns
    user_predicted_tracks = popular_tracks_df.join(user_relevant_tracks, on=['user_id'], how='inner')
    
    # Calculate the Mean Average Precision
    user_predicted_tracks_rdd = user_predicted_tracks.select("predicted", "actual").rdd
    ranking_metrics = RankingMetrics(user_predicted_tracks_rdd)
    map_score = ranking_metrics.meanAveragePrecisionAt(top_n)
    p_K = ranking_metrics.precisionAt(top_n)
    ncdg_K = ranking_metrics.ndcgAt(top_n)
    return map_score, p_K, ncdg_K

def ALS_model(spark,train_data, val_data, reg_param, rank_param, alpha_param, max_iter_param):

    # Create ALS model
    als = ALS(  rank= rank_param, maxIter= max_iter_param, regParam = reg_param, alpha=alpha_param,
                seed = 1, userCol="user_id", itemCol="track_id_index", ratingCol="play_count",
                coldStartStrategy="drop", nonnegative=True, implicitPrefs = True) #, checkpointInterval = 10)

    # Define evaluator as RMSE
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="play_count", predictionCol="prediction")

    #Fit ALS model to training data
    best_model = als.fit(train_data)

    # Generate predictions and evaluate using RMSE for best model
    predictions = best_model.transform(val_data)
    rmse = evaluator.evaluate(predictions) 
    precision_at_K, mean_average_precision, ncdg_at_K = evaluation(predictions, best_model, validation_data=val_data, top_n=100)

    print(f"RMSE score: {rmse}, MAP score: {mean_average_precision}, Precision at K: {precision_at_K}, NDCG at K: {ncdg_at_K}")

    return rmse, mean_average_precision,precision_at_K, ncdg_at_K, best_model

def test_data_results(test_data, best_model):
    
    predictions = best_model.transform(test_data)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="play_count", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    precision_at_K, mean_average_precision, ncdg_at_K = evaluation(predictions, best_model, validation_data=test_data, top_n=100)
    print("Test data results:")
    print(f"RMSE score: {rmse}, MAP score: {mean_average_precision}, Precision at K: {precision_at_K}, NDCG at K: {ncdg_at_K}")

    return rmse, mean_average_precision,precision_at_K, ncdg_at_K


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName("Recommender-DataLoader-GRP7").getOrCreate() # .config("spark.kryoserializer.buffer.max", "350m").getOrCreate()
    logger = spark._jvm.org.apache.log4j
    logger.LogManager.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(logger.Level.ERROR)
    host = "nyu-dataproc-m"
    small = True if sys.argv[1] == "small" else False
    net_id = getpass.getuser()
    play_count_threshold = 30

    # Check if we are using the small dataset or the full dataset
    if small:
        print(f"Using small data set")
        size='small'
        train_file =  f'/user/{net_id}/ListenBrainz/train_small.parquet'
        validation_file = f'/user/{net_id}/ListenBrainz/validation_small.parquet'
        res_file = f'/user/{net_id}/ListenBrainz/results_small.csv'
        
    else:
        print(f"Using full data set")
        size='full'
        train_file =  f'/user/{net_id}/ListenBrainz/train.parquet'
        validation_file = f'/user/{net_id}/ListenBrainz/validation.parquet'
        test_file = f'/user/{net_id}/ListenBrainz/test.parquet'
        res_file = f'/user/{net_id}/ListenBrainz/results.csv'

    train_data = spark.read.parquet(train_file, header=True,  schema='user_id INT, recording_msid STRING, play_count INT')
    train_data = train_data.dropDuplicates(["user_id", "recording_msid", "play_count"])
    train_data.createOrReplaceTempView(f'train_data')

    validation_data = spark.read.parquet(validation_file, header=True, schema='user_id INT, recording_msid STRING, play_count INT')
    validation_data = validation_data.dropDuplicates(["user_id", "recording_msid", "play_count"])
    validation_data.createOrReplaceTempView(f'validation_data')

    # Clean the data
    train_data = train_data.filter(train_data.play_count >= play_count_threshold)
    validation_data = validation_data.filter(validation_data.play_count >= play_count_threshold)

    stringIndexer = StringIndexer(inputCol="recording_msid", outputCol="track_id_index", handleInvalid="keep")
    model = stringIndexer.fit(train_data)
    indexed_train = model.transform(train_data)
    indexed_train = indexed_train.drop("recording_msid")
    indexed_train.persist()

    stringIndexer_val = StringIndexer(inputCol="recording_msid", outputCol="track_id_index", handleInvalid="keep")
    model_val = stringIndexer_val.fit(validation_data)
    indexed_val = model_val.transform(validation_data)
    indexed_val = indexed_val.drop("recording_msid")
    indexed_val.persist()

    # rank_param_list = [ 10, 50, 100, 150, 200]
    # reg_param_list = [0.01, 0.05 , 0.1, 0.2]
    # alpha_param_list = [0.1,1, 5, 10,]
    # max_iter_param_list = [5, 10, 15, 20]

    rank_param_list = [10]
    reg_param_list = [0.05]
    alpha_param_list = [0.1]
    max_iter_param_list = [20]
    param_combinations = itertools.product(rank_param_list, reg_param_list, alpha_param_list, max_iter_param_list)

    res_list = []

    loop_start_time = time.time()

    for rank, reg_param, alpha, max_iter in param_combinations:
    # Call function to create count matrix
        start_time = time.time()
        print(f"Starting ALS with rank: {rank}, reg_param: {reg_param}, alpha: {alpha}, max_iter: {max_iter}")
        rmse, mean_average_precision, precision_at_K, ncdg_at_K, best_model = ALS_model(spark, indexed_train, indexed_val, reg_param, rank, alpha, max_iter)
        res_list.append([rank, reg_param, alpha, max_iter, rmse, mean_average_precision, precision_at_K, ncdg_at_K ])
        print(f"ALS Dataset Size: {size}, Time: {time.time() - start_time} seconds \n")

    if not small:
        train_data.unpersist()
        validation_data.unpersist()
        test_data = spark.read.parquet(test_file, header=True, schema='user_id INT, recording_msid STRING, play_count INT')
        test_data = test_data.dropDuplicates(["user_id", "recording_msid", "play_count"])
        test_data.createOrReplaceTempView(f'test_data')
        test_data = test_data.filter(test_data.play_count >= play_count_threshold)
        test_data = model.transform(test_data)
        test_data = test_data.drop("recording_msid")
        test_data.persist()
        rmse, mean_average_precision, precision_at_K, ncdg_at_K = test_data_results(test_data, best_model)
    spark.stop()
