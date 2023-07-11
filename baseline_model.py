# Import command line arguments and helper functions
import sys
import numpy as np
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession, types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import broadcast, col, count, desc, row_number, avg, udf, countDistinct,expr, sum as spark_sum, abs as spark_abs
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics

def split_data_by_user(df):
    """
    Split the data into training and validation sets with an 80/20 ratio for each user.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame to split.
        
    Returns
    -------
    train_data, validation_data : DataFrame, DataFrame
        The training and validation DataFrames.
    """

    # Define a window partitioned by user_id and ordered by timestamp
    user_window = Window.partitionBy("user_id").orderBy("timestamp")

    # Add a row number column to the input DataFrame within each user partition
    df = df.withColumn("row_number", row_number().over(user_window))

    # Calculate the total number of rows per user partition
    user_counts = df.groupBy("user_id").agg(count("*").alias("total_rows"))

    # Join the input DataFrame with user_counts on user_id
    df = df.join(user_counts, on="user_id", how="inner")

    # Calculate the 80/20 split threshold for each user
    df = df.withColumn("split_threshold", (col("total_rows") * 0.8).cast("integer"))

    # Split data into training and validation sets based on the row number and split threshold
    train_data = df.where(col("row_number") <= col("split_threshold")).drop("row_number", "total_rows", "split_threshold")
    validation_data = df.where(col("row_number") > col("split_threshold")).drop("row_number", "total_rows", "split_threshold")

    return train_data, validation_data

def process_song_data(spark,net_id, small=False):
    '''Loads ListenBrainz data, processes them into dataframe and divide into train and validation testset
    Parameters
    ----------
    spark : SparkSession object
    small : whether to use the small dataset or the full dataset
    '''

    test_users_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_test.parquet'
    test_tracks_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_test.parquet'
    test_interactions_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet'
    test_file=f'/user/{net_id}/ListenBrainz/test_baseline.parquet'

    # Check if we are using the small dataset or the full dataset
    if small:
        print(f"Using small data set")
        interactions_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet'
        tracks_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet'
        users_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_train_small.parquet'
        train_file =  f'/user/{net_id}/ListenBrainz/train_small_baseline.parquet'
        validation_file = f'/user/{net_id}/ListenBrainz/validation_small_baseline.parquet'
        
    else:
        print(f"Using full data set")
        interactions_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet'
        tracks_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet'
        users_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_train.parquet'
        train_file =  f'/user/{net_id}/ListenBrainz/train_baseline.parquet'
        validation_file = f'/user/{net_id}/ListenBrainz/validation_baseline.parquet'

    # Check if data already exists
    try:
        train_data = spark.read.parquet(train_file)
        validation_data = spark.read.parquet(validation_file)
        test_interactions = spark.read.parquet(test_file)
        print("Data found, loading data")
        return train_data, validation_data, interactions, tracks, test_interactions
    except:
        print("No data found, processing data")
        pass
    
    print("Loading Interactions and tracks data")
    # Load the ListenBrainz TRAIN data
    interactions = spark.read.parquet(interactions_file, header=True,schema='user_id INT, recording_msid STRING nullable=false, timestamp INT')
    interactions.createOrReplaceTempView('interactions')
    tracks = spark.read.parquet(tracks_file, header=True, schema='recording_msid STRING, artist_name STRING, track_name STRING, recording_mbid STRING')
    tracks.createOrReplaceTempView('tracks')
    # users = spark.read.parquet(users_file, header=True,schema='user_id INT, user_name STRING')
    # users.createOrReplaceTempView('users') 
    print("Loading Test data")
    # Load the ListenBrainz TEST data
    test_interactions = spark.read.parquet(test_interactions_file, header=True, schema='user_id INT, recording_msid STRING nullable=false, timestamp INT')
    test_interactions.createOrReplaceTempView('test_interactions')
    test_tracks = spark.read.parquet(test_tracks_file, header=True, schema='recording_msid STRING, artist_name STRING, track_name STRING, recording_mbid STRING')
    test_tracks.createOrReplaceTempView('test_tracks')
    # test_users = spark.read.parquet(test_users_file, header=True, schema='user_id INT, user_name STRING')
    # test_users.createOrReplaceTempView('test_users')     
    
    print("Processing data")
    # Join interactions with tracks and users DataFrames


    # Drop the recording_mbid column
    interactions = interactions.drop("recording_mbid")

    test_interactions = test_interactions.drop("recording_mbid")

    # Drop rows with null values
    interactions = interactions.dropna()
    test_interactions = test_interactions.dropna()

    print("Splitting data")
    # Split data into training and validation sets
    train_data, validation_data = split_data_by_user(interactions) #complete_data.randomSplit([0.8, 0.2], seed=42)
    train_data.createOrReplaceTempView('train_data')
    validation_data.createOrReplaceTempView('validation_data')

    print("Persisting data") 

    # print("Saving train data")
    # train_data.write.mode('overwrite').parquet(train_file)
    # print("Saving validation data")
    # validation_data.write.mode('overwrite').parquet(validation_file)
    print("Saving test data")
    test_interactions.write.mode('overwrite').parquet(test_file)

    return train_data,validation_data,interactions, tracks, test_interactions

def popularity_baseline_model(data, top_n=100, damping_factor = 100, user_threshold=30):
    '''Returns the top n tracks in the data by average listen per user
    Parameters
    ----------
    data : DataFrame containing the data
    tracks : DataFrame containing the tracks
    top_n : number of top tracks to recommend
    '''

    # Calculate the number of listens and unique users per track
    track_counts = data.groupBy("recording_msid").agg(count("user_id").alias("listen_count"), countDistinct("user_id").alias("user_count"))
    track_counts.createOrReplaceTempView('track_counts')

    # Remove tracks with less than 30 users
    track_counts = track_counts.where(col("user_count") >= user_threshold)
    
    # Calculate the average listen per user for each track
    track_counts = track_counts.withColumn("avg_listen_per_user", col("listen_count") / (col("user_count") + damping_factor))
    track_counts.createOrReplaceTempView('track_counts')

    # Sort by avg_listen_per_user in descending order
    sorted_popularity = track_counts.sort(desc("avg_listen_per_user"))

    # Return the top n tracks
    top_tracks_general = sorted_popularity.limit(top_n)

    return top_tracks_general


def evaluation(popular_track_ids, validation_data, top_n):
    '''Returns the Mean Average Precision and Precision at K for the popular tracks baseline model
    Parameters
    ----------
    popular_tracks_df : DataFrame containing the top n tracks
    validation_data : DataFrame containing the validation data
    top_n : number of top tracks to recommend
    '''
    print("Evaluating model")
    # Get the list of popular track IDs
    # popular_track_ids = popular_tracks_df.select('recording_msid').rdd.flatMap(lambda x: x).collect()

    print("Grouping data")
    # Group the validation_data by user_id and collect their recording_msid into a list
    user_relevant_tracks = validation_data.groupBy('user_id').agg(F.collect_list('recording_msid').alias('actual'))

    print("Creating predictions")
    # Create a new DataFrame with a 'predicted' column containing the list of popular tracks for each user
    user_predicted_tracks = user_relevant_tracks.withColumn('predicted', F.array([F.lit(track_id) for track_id in popular_track_ids[:top_n]]))

    # Calculate the Mean Average Precision
    print("RDD")
    user_predicted_tracks_rdd = user_predicted_tracks.select("predicted", "actual").rdd
    ranking_metrics = RankingMetrics(user_predicted_tracks_rdd)
    print("Calculating MAP")
    map_score = ranking_metrics.meanAveragePrecisionAt(top_n)
    print("Calculating Precision at K")
    p_K = ranking_metrics.precisionAt(top_n)
    print("Calculating NDCG at K")
    ncdg_K = ranking_metrics.ndcgAt(top_n)
    return map_score, p_K, ncdg_K
    #pass

def popularity_baseline(spark, train_data=None, validation_data=None,test_data= None,tracks=None, top_n = 100, damping_factor = 100):
    '''Creates a general popularity baseline model and evaluates it on the validation set
        We use the number of times a track has been listened to as the popularity metric
        We calculate the popularity of each track in the training set and rank them in descending order

    Parameters
    ----------
    spark : SparkSession object
    train_data : training set
    validation_data : validation set
    tracks : tracks DataFrame
    users : users DataFrame
    top_n : number of top tracks to recommend
    create_user_popularity : whether to create a popularity baseline per user
    '''

    print("we are starting popularity baseline")
    # Get the top n tracks in the training set by popularity
    top_tracks_general_train = popularity_baseline_model(train_data, top_n, damping_factor = damping_factor)
    print(f"We finished baseline 1")
    train_data.unpersist()
    popular_track_ids = top_tracks_general_train.select('recording_msid').rdd.flatMap(lambda x: x).collect()
    map_score,pk,ncdg = evaluation(popular_track_ids, validation_data,top_n)
    print("For validation data")
    print(f"Damping factor: {damping_factor}")
    print(f"Mean Average Precision: {map_score}")
    print(f"pk: {pk}")
    print(f"ncdg: {ncdg}")
    map_score,pk,ncdg = evaluation(popular_track_ids, test_data,top_n)
    print("For test data")
    print(f"Damping factor: {damping_factor}")
    print(f"Mean Average Precision: {map_score}")
    print(f"pk: {pk}")
    print(f"ncdg: {ncdg}")

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder\
            .appName('Recommender-DataLoader-GRP7')\
            .config("spark.executor.memory", "20g")\
            .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    host = "nyu-dataproc-m"
    small = True if sys.argv[1] == "small" else False
    netID = getpass.getuser()
    train_data, validation_data, interactions, tracks, test_data = process_song_data(spark, netID, small=small)

    popularity_baseline(spark,train_data=train_data,validation_data=validation_data, test_data=test_data,tracks=tracks, top_n = 100, damping_factor = 200)

    spark.stop()

