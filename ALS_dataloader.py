
import getpass
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, row_number
from pyspark.sql.window import Window


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
    # Check if we are using the small dataset or the full dataset
    if small:
        print(f"Using small data set")
        interactions_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet'
        train_file =  f'/user/{net_id}/ListenBrainz/train_small.parquet'
        validation_file = f'/user/{net_id}/ListenBrainz/validation_small.parquet'
        
    else:
        print(f"Using full data set")
        interactions_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet'
        train_file =  f'/user/{net_id}/ListenBrainz/train.parquet'
        validation_file = f'/user/{net_id}/ListenBrainz/validation.parquet'

    # Check if data already exists
    try:
        train_data = spark.read.parquet(train_file, header=True, schema='user_id INT, recording_msid STRING, play_count INT')
        validation_data = spark.read.parquet(validation_file, header=True, schema='user_id INT, recording_msid STRING, play_count INT')
        print("Data found, loading data")
        return train_data, validation_data, interactions
    except Exception as e:
        print(e)
        print("No data found, processing data")
        pass

    # Load the ListenBrainz TRAIN data
    interactions = spark.read.parquet(interactions_file, header=True,schema='user_id INT, recording_msid STRING nullable=false, timestamp INT')
    interactions.createOrReplaceTempView('interactions')

    # Drop the recording_mbid column
    interactions = interactions.drop("recording_mbid")

    # Drop rows with null values
    interactions = interactions.dropna()
    
    # Split data into training and validation sets
    train_data, validation_data = split_data_by_user(interactions) #interactions.randomSplit([0.8, 0.2], seed=42)
    interactions.unpersist()

    write_data(train_data, train_file)
    write_data(validation_data, validation_file)



def write_data(data, file):

    print('calculating playcount')
    play_count = data.groupBy("user_id", "recording_msid").agg(count("*").alias("play_count"))

    indexed_with_play_count = data.join(play_count, on=["user_id", "recording_msid"], how="inner")

    indexed_with_play_count = indexed_with_play_count.fillna(0, subset=["play_count"])

    indexed_with_play_count = indexed_with_play_count.withColumn("play_count", col("play_count").cast("int"))

    indexed_with_play_count.printSchema()
    indexed_with_play_count.write.mode('overwrite').parquet(file)
    indexed_with_play_count.unpersist()

def get_test_data(spark, net_id, small=False):

    test_interactions_file = 'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet'
    test_file=f'/user/{net_id}/ListenBrainz/test.parquet'

    test_interactions = spark.read.parquet(test_interactions_file, header=True, schema='user_id INT, recording_msid STRING nullable=false, timestamp INT')
    test_interactions.createOrReplaceTempView('test_interactions')


    test_interactions = test_interactions
    test_interactions = test_interactions.drop("recording_mbid")
    test_interactions.createOrReplaceTempView('test_interactions')


    play_count = test_interactions.groupBy("user_id", "recording_msid").agg(count("*").alias("play_count"))

    indexed_with_play_count = test_interactions.join(play_count, on=["user_id", "recording_msid"], how="inner")

    indexed_with_play_count = indexed_with_play_count.fillna(0, subset=["play_count"])
    
    indexed_with_play_count = indexed_with_play_count.withColumn("play_count", col("play_count").cast("int"))

    indexed_with_play_count.printSchema()
    indexed_with_play_count.write.mode('overwrite').parquet(test_file)

    indexed_with_play_count.unpersist()

    return test_interactions

    return test_data
if __name__ == "__main__":

    spark = SparkSession.builder\
            .appName('Recommender-DataLoader-GRP7')\
            .config("spark.executor.memory", "20g")\
            .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    host = "nyu-dataproc-m"
    small = True if sys.argv[1] == "small" else False
    netID = getpass.getuser()
    process_song_data(spark, netID, small=small)
    if not small:
        test_data = get_test_data(spark, netID, small=small)
